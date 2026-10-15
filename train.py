"""Train DVSRC-Net on aligned user-level features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from filelock import FileLock

from dvsrc_net.data import load_training_inputs
from dvsrc_net.engine import run_train
from dvsrc_net.utils import atomic_json, file_hash, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.json")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_config(args.config)
    seeds = args.seeds if args.seeds is not None else config.pop("seeds")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct integers")
    config.pop("seeds", None)
    config.pop("selection_metric")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to use the CPU")
    data_root = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output == Path(__file__).parent.resolve():
        raise ValueError("Use a separate directory for training outputs")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".training.lock"), timeout=0):
        data, audit = load_training_inputs(data_root, None)
        source_root = Path(__file__).parent / "dvsrc_net"
        source_hashes = {p.name: file_hash(p) for p in sorted(source_root.glob("*.py"))}
        plan = {"seeds": seeds, "config": config, "input_audit": audit, "source_hashes": source_hashes}
        plan_path = output / "run_config.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("The output directory belongs to different data, code, or settings")
        atomic_json(plan_path, plan)
        records = []
        for seed in seeds:
            records.append(run_train(output, None, data, audit, "B+M+D", seed, config, device))
        atomic_json(output / "TRAINING_COMPLETE.json", {"seeds": seeds, "runs": records})
    print(f"Training complete: {output}")


if __name__ == "__main__":
    main()
