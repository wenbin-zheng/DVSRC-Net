"""Evaluate trained checkpoints using their validation-selected thresholds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from filelock import FileLock

from dvsrc_net.data import load_training_inputs
from dvsrc_net.engine import predict
from dvsrc_net.metrics import score, summarize
from dvsrc_net.model import create_model
from dvsrc_net.utils import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    plan = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    if complete["seeds"] != plan["seeds"]:
        raise ValueError("Training completion metadata does not match its configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to use the CPU")
    with FileLock(str(run / ".training.lock"), timeout=0):
        data, audit = load_training_inputs(root, None)
        if audit != plan["input_audit"]:
            raise ValueError("Evaluation inputs differ from the training inputs")
        for seed in plan["seeds"]:
            folder = run / f"seed_{seed}"
            finished = json.loads((folder / "TRAINED.json").read_text(encoding="utf-8"))
            if finished["seed"] != seed or file_hash(folder / "best.pt") != finished["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint verification failed for seed {seed}")
        indices = np.flatnonzero(data["split"] == 2)
        with np.load(root / "labels/sealed/test_labels.sealed.npz", allow_pickle=False) as labels:
            if set(labels.files) != {"user_id", "label"}:
                raise ValueError("Test labels must contain exactly user_id and label")
            if not np.array_equal(labels["user_id"].astype(str), data["user_id"][indices]):
                raise ValueError("Test label identities or ordering do not match the features")
            y = labels["label"].astype(np.int8)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Test labels must be binary")
        data["label"][indices] = y
        rows = []
        for seed in plan["seeds"]:
            folder = run / f"seed_{seed}"
            finished = json.loads((folder / "TRAINED.json").read_text(encoding="utf-8"))
            model = create_model(plan["config"], data["bank"].shape[1], device)
            model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
            _, probability, _ = predict(model, data, indices, device)
            metrics = score(y, probability, float(finished["threshold"]))
            rows.append({"seed": seed, **metrics})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result = {"per_seed": rows, "summary": summarize(rows)}
        atomic_json(run / "metrics.json", result)
    for name in ("pr_auc_trapezoid", "auroc", "precision", "recall", "f1"):
        value = result["summary"][name]
        suffix = f" ± {100 * value['sample_sd']:.2f}" if value["sample_sd"] is not None else ""
        print(f"{name}: {100 * value['mean']:.2f}{suffix}")
    print(f"Saved: {run / 'metrics.json'}")


if __name__ == "__main__":
    main()
