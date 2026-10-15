"""Training, validation, checkpointing, and prediction."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch import nn
from torch.nn import functional as F

from .data import batch_iter
from .model import DVSRCNet, create_model
from .utils import atomic_json, file_hash


def make_model(core, arm, bank_dim, config, device):
    if arm != "B+M+D":
        raise ValueError("Only the complete DVSRC-Net is supported")
    return create_model(config, bank_dim, device)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def threshold_by_validation_f1(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(score))])


def development_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    return {"ap": float(average_precision_score(y, probability)),
            "f1": float(f1_score(y, probability >= threshold, zero_division=0)),
            "threshold": float(threshold)}


@torch.inference_mode()
def predict(model: DVSRCNet, data: dict, indices: np.ndarray, device: torch.device, diagnostic: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval(); probabilities=[]; targets=[]; gates=[]; alignment=[]
    for x, y in batch_iter(data, indices, 128):
        x = {k: v.to(device) for k, v in x.items()}
        logits, state = model(x["text"].float(), x["image"].float(), x["mask"].float(), x["q03"].float())
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy()); targets.append(y.numpy())
        gates.append(state["gate"].cpu().numpy())
        alignment.append((F.normalize(state["task"], dim=-1) * F.normalize(state["visual_task"], dim=-1)).sum(-1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities), {"gate": np.concatenate(gates), "task_visual_cosine": np.concatenate(alignment)}


def gradient_norms(model: nn.Module) -> dict:
    groups = {"backbone": model.backbone, "task_projection": model.task_projection,
              "bank_projection": model.bank_projection, "m_relation": model.m_relation,
              "visual_task_projection": model.visual_task_projection, "d_relation": model.d_relation,
              "final_head": model.final_head}
    result = {}
    for name, module in groups.items():
        pieces = [p.grad.detach().norm().item() ** 2 for p in module.parameters() if p.grad is not None]
        result[name] = float(sum(pieces) ** .5)
    return result


def run_train(study: Path, core, data: dict, audit: dict, arm: str, seed: int, config: dict, device: torch.device,
              diagnostic: str | None = None) -> dict:
    name = f"seed_{seed}"
    folder = study / name
    done = folder / "TRAINED.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    if (study / "MODEL_LOCK.json").exists():
        raise RuntimeError("MODEL_LOCK.json exists: training is prohibited after lock")
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = make_model(core, arm, data["bank"].shape[1], config, device)
    optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"], weight_decay=config["weight_decay"])
    train_indices, validation_indices = np.flatnonzero(data["split"] == 0), np.flatnonzero(data["split"] == 1)
    counts = np.bincount(data["label"][train_indices], minlength=2)
    weights = torch.tensor((counts.sum() / (2 * counts)) ** config["class_power"], dtype=torch.float32, device=device)
    atomic_json(folder / "config.json", {"arm": arm, "diagnostic": diagnostic, "seed": seed, "config": config,
                                            "selection": "checkpoint validation AP; threshold validation F1", "test_used": False,
                                            "input_audit": audit, "source_sha256": file_hash(Path(__file__))})
    start, best, best_epoch, history = 1, -float("inf"), 0, []
    resume = folder / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimiser.load_state_dict(state["optimiser"])
        start, best, best_epoch, history = state["epoch"] + 1, state["best"], state["best_epoch"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    for epoch in range(start, config["epochs"] + 1):
        model.train(); total_loss = 0.0; seen = 0; m_rows = d_rows = m_negative = 0; last_grad = {}
        for x, y in batch_iter(data, train_indices, config["batch"], seed + epoch * 100003):
            x = {k: v.to(device) for k, v in x.items()}; y = y.to(device).long()
            optimiser.zero_grad(set_to_none=True)
            logits, state = model(x["text"].float(), x["image"].float(), x["mask"].float(), x["q03"].float())
            ce = F.cross_entropy(logits, y, weight=weights, reduction="none")
            pt = logits.softmax(-1).gather(1, y[:, None]).squeeze(1)
            loss = ((1 - pt) ** config["focal_gamma"] * ce).mean()
            if arm in ("B+M", "B+M+D") or diagnostic == "SHUFFLE-M":
                m_loss, meta = model.semantic_loss(state["task"], torch.from_numpy(data["bank"]).to(device), x["social_top"], x["qwen_top"], x["social_query"].float())
                loss = loss + config["m_weight"] * m_loss; m_rows += meta["m_supported_rows"]; m_negative += meta["m_explicit_negatives"]
            if arm in ("B+D", "B+M+D") or diagnostic == "SHUFFLE-D":
                d_loss, meta = model.image_text_loss(state["task"], state["visual_task"], state["gate"], state["has_image"])
                loss = loss + config["d_weight"] * d_loss; d_rows += meta["d_supported_rows"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward(); last_grad = gradient_norms(model); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            total_loss += float(loss.detach()) * len(y); seen += len(y)
        yv, pv, diagnostics = predict(model, data, validation_indices, device, diagnostic)
        ap = float(average_precision_score(yv, pv))
        record = {"epoch": epoch, "loss": total_loss / max(seen, 1), "validation_ap": ap,
                  "m_supported_rows": m_rows, "m_explicit_negatives": m_negative, "d_supported_rows": d_rows,
                  "gradient_norm": last_grad, "gate": {"mean": float(diagnostics["gate"].mean()), "std": float(diagnostics["gate"].std()),
                  "p05": float(np.quantile(diagnostics["gate"], .05)), "p95": float(np.quantile(diagnostics["gate"], .95))},
                  "task_visual_alignment": {"mean": float(diagnostics["task_visual_cosine"].mean()), "std": float(diagnostics["task_visual_cosine"].std())}}
        print(f"Epoch {epoch}/{config['epochs']} | loss={record['loss']:.5f} | validation AP={ap:.6f} | {name}", flush=True)
        history.append(record)
        if ap > best:
            best, best_epoch = ap, epoch
            atomic_torch(folder / "best.pt", model.state_dict())
            np.savez_compressed(folder / "validation_predictions.npz", user_id=data["user_id"][validation_indices], label=yv, probability=pv)
        atomic_torch(resume, {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "epoch": epoch,
                              "best": best, "best_epoch": best_epoch, "history": history,
                              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        atomic_json(folder / "history.json", history)
        atomic_json(study / "status_train.json", {"stage": "training", "run": name, "epoch": epoch, "best_epoch": best_epoch, "updated": time.time()})
        if epoch >= config["min_epochs"] and epoch - best_epoch >= config["patience"]:
            break
    validation = np.load(folder / "validation_predictions.npz", allow_pickle=False)
    threshold = threshold_by_validation_f1(validation["label"], validation["probability"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {"arm": arm, "diagnostic": diagnostic, "seed": seed, "best_epoch": best_epoch, "actual_epochs": history[-1]["epoch"],
              "validation": development_metrics(validation["label"], validation["probability"], threshold), "threshold": threshold,
              "checkpoint_sha256": file_hash(folder / "best.pt"), "parameter_count": parameter_count,
              "trainable_parameter_count": trainable_count, "last_epoch_diagnostics": history[-1], "test_used": False}
    atomic_json(done, result)
    resume.unlink(missing_ok=True)
    del model, optimiser
    if device.type == "cuda": torch.cuda.empty_cache()
    return result
