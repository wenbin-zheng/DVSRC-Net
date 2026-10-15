"""Assemble multimodal and dual-view symptom retrieval inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from filelock import FileLock

from dvsrc_net.data import SPLIT_COUNTS
from dvsrc_net.features import encode, fit_encoder, make_document, query_texts, read_users, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    raw = root / "raw"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".features.lock"), timeout=0):
        with np.load(raw / "split_manifest.npz", allow_pickle=False) as manifest:
            uid = manifest["user_id"].astype(str)
            split = manifest["split"].astype(np.int8)
            label = manifest["label"].astype(np.int8)
        if len(uid) != len(np.unique(uid)) or tuple(np.bincount(split, minlength=3)) != SPLIT_COUNTS:
            raise ValueError("Unexpected user membership or split counts")
        if label.shape != uid.shape or not np.isin(label, [0, 1]).all():
            raise ValueError("Labels must be an aligned binary vector")
        with np.load(raw / "modalities.npz", allow_pickle=False) as modalities:
            if not np.array_equal(modalities["user_id"].astype(str), uid):
                raise ValueError("Modality user identities/order do not match the split manifest")
            text = modalities["text"].astype(np.float32)
            image = modalities["image"].astype(np.float32)
            mask = modalities["mask"].astype(np.float32)
        if text.shape != (len(uid), 11, 500) or image.shape != (len(uid), 512) or mask.shape != (len(uid),):
            raise ValueError("Expected text [N,11,500], image [N,512], mask [N]")
        users_path = raw / "users.jsonl.gz"
        if not users_path.exists():
            users_path = raw / "users.jsonl"
        rows = read_users(users_path)
        if [row["user_id"] for row in rows] != uid.tolist():
            raise ValueError("Text user identities/order do not match the split manifest")
        cache = args.query_cache if args.query_cache is not None else root / "cache/qwen_queries.sqlite3"
        queries = query_texts(rows, cache, args.qwen_model, args.device)
        encoder = fit_encoder(rows, uid, split, root / "cache/retrieval_encoder.joblib")
        resource = Path(__file__).parent / "resources/symptoms.json"
        concepts = json.loads(resource.read_text(encoding="utf-8"))["concepts"]
        descriptions = [description for concept in concepts for description in concept["descriptions"]]
        if len(concepts) != 11 or len(descriptions) != 44:
            raise ValueError("The symptom bank must have 11 concepts and 44 descriptions")
        social_query = encode(encoder, [make_document(row) for row in rows])
        qwen_query = encode(encoder, queries)
        bank = encode(encoder, descriptions, 44)
        social, social_top = retrieve(social_query, bank)
        qwen, qwen_top = retrieve(qwen_query, bank)
        q03 = np.concatenate((social, qwen, np.abs(social - qwen)), axis=1).astype(np.float32)
        features = root / "features"
        labels = root / "labels"
        features.mkdir(exist_ok=True)
        (labels / "sealed").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "membership.npz", user_id=uid, split=split)
        np.savez_compressed(labels / "dev_labels.npz", user_id=uid[split < 2], split=split[split < 2], label=label[split < 2])
        np.savez_compressed(labels / "sealed/test_labels.sealed.npz", user_id=uid[split == 2], label=label[split == 2])
        np.savez_compressed(features / "medical_dualspace_unlabeled.npz", user_id=uid, text=text,
                            image=image, mask=mask, Q03=q03, social_query=social_query,
                            qwen_query=qwen_query, social_retrieval=social, qwen_retrieval=qwen,
                            medical_bank=bank, social_top=social_top, qwen_top=qwen_top)
    print(f"Features prepared: {features}")


if __name__ == "__main__":
    main()
