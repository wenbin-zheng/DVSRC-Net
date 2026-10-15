"""DVSRC-Net model."""

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Model


class DVSRCNet(nn.Module):
    """Dual-view retrieval, DCIM, and both contrastive objectives."""
    def __init__(self, core_model, bank_dim: int, config: dict):
        super().__init__()
        self.backbone = core_model
        self.config = config
        self.text_reference = copy.deepcopy(core_model.text_encoder).eval()
        self.visual_reference = copy.deepcopy(core_model.image_encoder).eval()
        for module in (self.text_reference, self.visual_reference):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.task_projection = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.Tanh())
        self.bank_projection = nn.Sequential(nn.Linear(bank_dim, 128), nn.LayerNorm(128))
        self.q03_projection = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.Tanh())
        self.m_relation = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(128, 128))
        self.visual_task_projection = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.Tanh())
        self.content_gate_scale = nn.Parameter(torch.tensor(3.0))
        self.content_gate_bias = nn.Parameter(torch.tensor(0.0))
        self.d_relation = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(192, 128))
        self.final_head = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(256, 2))

    @staticmethod
    def _normal(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-8)

    def forward(self, text, image, mask, q03):
        with torch.no_grad():
            reference_text = self.text_reference(text)
            reference_visual = self.visual_reference(image * mask[:, None]) * mask[:, None]
        task_text_raw = self.backbone.text_encoder(text)
        image_hidden = self.backbone.image_encoder(image * mask[:, None]) * mask[:, None]
        task = self.task_projection(task_text_raw)
        m_context = self.q03_projection(q03)
        task = task + self.m_relation(torch.cat((task, m_context), dim=-1))
        visual_task = self.visual_task_projection(image_hidden) * mask[:, None]
        content_similarity = (self._normal(reference_text) * self._normal(reference_visual)).sum(-1)
        gate = torch.sigmoid(self.content_gate_scale * content_similarity + self.content_gate_bias) * mask
        relation = self.d_relation(torch.cat((task, visual_task, (task-visual_task).abs()), dim=-1))
        task = task + gate[:, None] * relation
        logits = self.final_head(torch.cat((task, image_hidden), dim=-1))
        state = {"task": task, "visual_task": visual_task, "reference_text": reference_text.detach(),
                 "gate": gate, "m_context": m_context, "has_image": mask}
        return logits, state

    def semantic_loss(self, task: torch.Tensor, bank: torch.Tensor, social_top: torch.Tensor,
                    qwen_top: torch.Tensor, social_query: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Evidence-constrained multi-positive contrast. Unsupported bank items vanish.

        Positives are exactly original-social/Qwen top-index intersections.  The
        denominator contains positives plus explicit negatives only: bottom
        social-similarity bank entries below a fixed threshold.  Rows without a
        supported positive are excluded, not treated as negatives.
        """
        prototypes = self._normal(self.bank_projection(bank))
        query = self._normal(task)
        social = self._normal(social_query)
        original_bank = self._normal(bank)
        social_similarity = social @ original_bank.T
        losses, supported, negative_total = [], 0, 0
        for row in range(query.shape[0]):
            s = set(int(i) for i in social_top[row].tolist() if int(i) >= 0)
            q = set(int(i) for i in qwen_top[row].tolist() if int(i) >= 0)
            positive = sorted(s.intersection(q))
            if not positive:
                continue
            candidates = set(positive)
            ranked_negative = torch.argsort(social_similarity[row]).tolist()
            negatives = [int(i) for i in ranked_negative if int(i) not in candidates and float(social_similarity[row, i]) < self.config["negative_similarity_threshold"]][:self.config["negative_k"]]
            candidates.update(negatives)
            candidate = torch.tensor(sorted(candidates), device=query.device, dtype=torch.long)
            positive_mask = torch.tensor([i in positive for i in candidate.tolist()], device=query.device, dtype=torch.bool)
            logits = (query[row:row+1] @ prototypes[candidate].T).flatten() / self.config["temperature"]
            losses.append(torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positive_mask], dim=0))
            supported += 1
            negative_total += len(negatives)
        if not losses:
            return task.sum() * 0.0, {"m_supported_rows": 0, "m_explicit_negatives": 0, "m_objective": "no supported positives in batch"}
        return torch.stack(losses).mean(), {"m_supported_rows": supported, "m_explicit_negatives": negative_total,
                                             "m_objective": "evidence-constrained supervised multi-positive logsumexp"}

    def image_text_loss(self, task: torch.Tensor, visual_task: torch.Tensor, gate: torch.Tensor,
                  has_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        eligible = (has_image > 0) & (gate >= self.config["d_gate_floor"])
        if int(eligible.sum()) < 2:
            return task.sum() * 0.0, {"d_supported_rows": int(eligible.sum()), "d_objective": "insufficient gated pairs"}
        a, b = self._normal(task[eligible]), self._normal(visual_task[eligible])
        logits = a @ b.T / self.config["temperature"]
        # Same-user relation is the positive. The detached content gate chose the
        # rows; its reference spaces remain frozen, avoiding visual symptom claims.
        targets = torch.arange(len(a), device=a.device)
        return F.cross_entropy(logits, targets), {"d_supported_rows": int(eligible.sum()),
                                                    "d_objective": "content-gated same-user relation contrast"}


def create_model(config: dict, bank_dim: int = 256, device="cpu") -> DVSRCNet:
    backbone_config = dict(mode="gated", dropout=config["dropout"], head_dropout=0.0,
                           norm="group", bn_momentum=.01, head="linear", modality_drop=0.0)
    return DVSRCNet(Model(backbone_config), bank_dim, config).to(device)
