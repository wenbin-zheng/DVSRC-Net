"""Aligned feature inputs and train/validation label isolation."""

from pathlib import Path
import numpy as np
import torch

from .utils import file_hash, tensor_hash

SPLIT_COUNTS = (18588, 2324, 2325)


def _get(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError("missing one of " + ", ".join(names) + "; available=" + ", ".join(npz.files))


def discover_feature_file(study: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied
    options = (study / "features" / "medical_dualspace_unlabeled.npz",
               study / "features" / "qwen_controls.npz",
               study / "features" / "dualspace_features_unlabeled.npz")
    for item in options:
        if item.is_file():
            return item
    raise FileNotFoundError("Pass --features; expected a label-free archive with social/qwen views and medical bank")


def load_training_inputs(study: Path, supplied_features: Path | None) -> tuple[dict, dict]:
    """Open only whitelisted development inputs; enforce all identity and split invariants."""
    membership_path = study / "membership.npz"
    labels_path = study / "labels" / "dev_labels.npz"
    features_path = discover_feature_file(study, supplied_features)
    with np.load(membership_path, allow_pickle=False) as membership, \
         np.load(labels_path, allow_pickle=False) as development, \
         np.load(features_path, allow_pickle=False) as feature:
        if set(membership.files) != {"user_id", "split"}:
            raise RuntimeError("membership must contain exactly user_id and split")
        if set(development.files) != {"user_id", "split", "label"}:
            raise RuntimeError("development labels must contain exactly user_id, split, label")
        if any("label" in name.lower() for name in feature.files):
            raise RuntimeError("feature archive must be label-free")
        user_id = membership["user_id"].astype(str)
        split = membership["split"].astype(np.int8)
        if len(user_id) != sum(SPLIT_COUNTS) or tuple(np.bincount(split, minlength=3)) != SPLIT_COUNTS:
            raise RuntimeError(f"required fixed split is {SPLIT_COUNTS}, found {tuple(np.bincount(split, minlength=3))}")
        if len(np.unique(user_id)) != len(user_id) or not np.isin(split, [0, 1, 2]).all():
            raise RuntimeError("invalid membership identities or split values")
        feature_ids = _get(feature, "user_id").astype(str)
        if not np.array_equal(user_id, feature_ids):
            raise RuntimeError("feature user_id order differs from membership")
        position = {value: index for index, value in enumerate(user_id)}
        y = np.full(len(user_id), -1, dtype=np.int8)
        for uid, group, label in zip(development["user_id"].astype(str), development["split"], development["label"]):
            index = position.get(uid)
            if index is None or split[index] != group or group == 2:
                raise RuntimeError("development labels include a missing or sealed-test user")
            y[index] = label
        if np.any(y[split < 2] < 0) or np.any(y[split == 2] >= 0):
            raise RuntimeError("development-label completeness/sealed-test isolation violation")
        text = _get(feature, "text").astype(np.float32)
        image = _get(feature, "image").astype(np.float32)
        mask = _get(feature, "mask").astype(np.float32)
        social_query = _get(feature, "social_query", "social").astype(np.float32)
        qwen_query = _get(feature, "qwen_query", "clinical_query", "qwen").astype(np.float32)
        social_retrieval = _get(feature, "social_retrieval", "social").astype(np.float32)
        qwen_retrieval = _get(feature, "qwen_retrieval", "clinical", "qwen").astype(np.float32)
        q03 = _get(feature, "Q03", "DV-Q03", "q03_retrieval") if any(k in feature.files for k in ("Q03", "DV-Q03", "q03_retrieval")) else np.concatenate((social_retrieval, qwen_retrieval, np.abs(social_retrieval-qwen_retrieval)), axis=1)
        bank = _get(feature, "medical_bank", "bank").astype(np.float32)
        social_top = _get(feature, "social_top").astype(np.int64)
        qwen_top = _get(feature, "qwen_top", "clinical_top").astype(np.int64)
    expected = len(user_id)
    shapes = {"text": (expected, 11, 500), "image": (expected, 512), "mask": (expected,),
              "social_query": (expected, social_query.shape[1]), "qwen_query": (expected, qwen_query.shape[1]),
              "social_retrieval": (expected, social_retrieval.shape[1]), "qwen_retrieval": (expected, qwen_retrieval.shape[1]),
              "q03": (expected, 768), "social_top": (expected, social_top.shape[1]), "qwen_top": (expected, qwen_top.shape[1])}
    values = dict(text=text, image=image, mask=mask, social_query=social_query, qwen_query=qwen_query,
                  social_retrieval=social_retrieval, qwen_retrieval=qwen_retrieval, q03=q03.astype(np.float32),
                  bank=bank, social_top=social_top, qwen_top=qwen_top)
    for name, expected_shape in shapes.items():
        if values[name].shape != expected_shape:
            raise RuntimeError(f"{name} expected {expected_shape}, found {values[name].shape}")
    if bank.ndim != 2 or bank.shape[1] != social_query.shape[1] or not (np.isfinite(bank).all() and all(np.isfinite(x).all() for x in values.values() if x.dtype.kind == "f")):
        raise RuntimeError("nonfinite or dimensionally incompatible feature values")
    if np.any((social_top < -1) | (social_top >= len(bank))) or np.any((qwen_top < -1) | (qwen_top >= len(bank))):
        raise RuntimeError("retrieval top index is outside bank range")
    data = dict(user_id=user_id, split=split, label=y, **values)
    audit = {"training_inputs": [str(membership_path), str(labels_path), str(features_path)],
             "forbidden_input": "labels/sealed/test_labels.sealed.npz", "split_counts": list(SPLIT_COUNTS),
             "membership_sha256": file_hash(membership_path), "dev_labels_sha256": file_hash(labels_path),
             "features_sha256": file_hash(features_path), "identity_sha256": tensor_hash(user_id.astype("S")),
             "features_are_label_free": True}
    return data, audit


def batch_iter(data: dict, indices: np.ndarray, batch_size: int, seed: int | None = None):
    order = np.asarray(indices)
    if seed is not None:
        order = np.random.default_rng(seed).permutation(order)
    fields = ("text", "image", "mask", "q03", "social_query", "social_top", "qwen_top")
    for start in range(0, len(order), batch_size):
        take = order[start:start + batch_size]
        yield {key: torch.from_numpy(data[key][take]) for key in fields}, torch.from_numpy(data["label"][take])
