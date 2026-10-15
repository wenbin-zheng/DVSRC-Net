"""Small file and configuration helpers."""

import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seeds", "batch", "epochs", "min_epochs", "patience", "lr", "weight_decay",
                "dropout", "focal_gamma", "class_power", "temperature", "m_weight", "d_weight",
                "negative_k", "negative_similarity_threshold", "d_gate_floor", "selection_metric"}
    if set(config) != required:
        raise ValueError(f"Unexpected configuration keys: {set(config) ^ required}")
    if config["selection_metric"] != "ap":
        raise ValueError("Unsupported checkpoint selection metric")
    if not 1 <= config["min_epochs"] <= config["epochs"] or config["patience"] < 1:
        raise ValueError("Invalid epoch or early-stopping settings")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be a nonempty list without duplicates")
    return config
