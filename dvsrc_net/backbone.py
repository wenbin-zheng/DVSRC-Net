"""FCN backbone and checkpoint-compatible module initialization."""

import torch
from torch import nn


class DSTSFCN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 11
        for output_channels, kernel in ((128, 8), (256, 5), (128, 3)):
            layers.extend([
                nn.Conv1d(channels, output_channels, kernel, padding="same"),
                nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
            channels = output_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1] != 11:
            raise ValueError("DSTS 输入必须为 [用户数, 11, 帖子数]，不能输入 TF-IDF")
        return self.layers(sequence).mean(dim=-1)


class LowRankFusion(nn.Module):
    def __init__(self, text_dim=128, image_dim=128, rank=4, output_dim=2):
        super().__init__()
        self.text_factor = nn.Parameter(torch.empty(rank, text_dim + 1, output_dim))
        self.image_factor = nn.Parameter(torch.empty(rank, image_dim + 1, output_dim))
        self.rank_weights = nn.Parameter(torch.empty(rank))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.image_factor)
        nn.init.normal_(self.rank_weights, std=rank ** -0.5)

    def forward(self, text, image):
        text_aug = torch.cat((text.new_ones(text.shape[0], 1), text), dim=-1)
        image_aug = torch.cat((image.new_ones(image.shape[0], 1), image), dim=-1)
        text_low = torch.einsum("bi,rio->bro", text_aug, self.text_factor)
        image_low = torch.einsum("bi,rio->bro", image_aug, self.image_factor)
        return torch.einsum("bro,r->bo", text_low * image_low, self.rank_weights) + self.bias


class SWDDModel(nn.Module):
    def __init__(self, mode="lmf", rank=4, dropout=0.3):
        super().__init__()
        if mode not in {"lmf", "concat", "text_only", "image_only"}:
            raise ValueError(mode)
        self.mode = mode
        self.text_encoder = DSTSFCN()
        self.image_encoder = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = LowRankFusion(rank=rank)
        self.concat_head = nn.Linear(256, 2)
        self.text_head = nn.Linear(128, 2)
        self.image_head = nn.Linear(128, 2)

    def forward(self, sequence, image, image_mask):
        mask = image_mask.reshape(-1, 1).to(image)
        # Mask before and after projection so arbitrary missing-image values cannot leak.
        image_hidden = self.image_encoder(image * mask) * mask
        text_hidden = self.text_encoder(sequence)
        if self.mode == "text_only":
            return self.text_head(text_hidden)
        if self.mode == "image_only":
            return self.image_head(image_hidden)
        if self.mode == "concat":
            return self.concat_head(torch.cat((text_hidden, image_hidden), dim=-1))
        return self.fusion(text_hidden, image_hidden)


class Model(SWDDModel):
    def __init__(self,c):
        super().__init__('concat',dropout=c['dropout'])
        self.c=c
        for i,m in enumerate(self.text_encoder.layers):
            if isinstance(m,nn.BatchNorm1d):
                if c['norm']=='group': self.text_encoder.layers[i]=nn.GroupNorm(8,m.num_features,eps=1e-3)
                else:m.momentum=c['bn_momentum']
        self.head_drop=nn.Dropout(c['head_dropout'])
        if c['mode'] in ('gated','residual_gate'):
            self.gate=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,128),nn.Sigmoid())
        if c['head']=='mlp':self.concat_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(c['head_dropout']),nn.Linear(128,2))
        if c['mode'].startswith('pair_'):
            self.post_text=nn.Sequential(nn.Linear(11,128),nn.ReLU())
            self.post_image=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(c['dropout']))
            self.post_joint=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(c['head_dropout']))
            self.post_score=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,1))

    def forward(self,t,v,m,p):
        th=self.text_encoder(t)
        mask=m[:,None]
        if self.training and self.c['modality_drop']:
            mask=mask*(torch.rand_like(mask)>self.c['modality_drop'])
        ih=self.image_encoder(v*mask)*mask
        mode=self.c['mode']
        if mode.startswith('pair_'):
            valid=p[:,:,523:524]
            if mode=='pair_mean':
                pooled=(p[:,:,11:523]*valid).sum(1)/valid.sum(1).clamp_min(1)
                pooled=nn.functional.normalize(pooled,dim=-1)
                pm=(valid.sum(1)>0).float()
                ih=self.image_encoder(pooled)*pm
            else:
                pt=self.post_text(p[:,:,:11]);pv=self.post_image(p[:,:,11:523])
                if mode=='pair_late':
                    pt=(pt*valid).sum(1)/valid.sum(1).clamp_min(1)
                    pv=(pv*valid).sum(1)/valid.sum(1).clamp_min(1)
                    ih=self.post_joint(torch.cat([pt,pv,pt*pv,(pt-pv).abs()],-1))*(valid.sum(1)>0)
                else:
                    h=self.post_joint(torch.cat([pt,pv,pt*pv,(pt-pv).abs()],-1))
                    scores=self.post_score(torch.cat([th[:,None,:].expand_as(h),h],-1))
                    a=scores.masked_fill(valid==0,-1e4).softmax(1)*valid
                    ih=(h*a).sum(1)/a.sum(1).clamp_min(1e-6)
        if mode in ('gated','residual_gate'):
            g=self.gate(torch.cat([th,ih],-1));ih=ih*(g if mode=='gated' else (0.5+g))
        return self.concat_head(self.head_drop(torch.cat([th,ih],-1)))
