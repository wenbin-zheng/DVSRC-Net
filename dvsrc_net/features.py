"""Label-free symptom-query generation and dual-view retrieval features."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def read_users(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [{"user_id": str(row["user_id"]), "texts": row.get("texts", [])} for row in rows]


def make_document(row):
    texts = [str(x).strip() for x in row.get("texts", []) if str(x).strip()]
    if not texts:
        return "无可用文本"
    per_post = max(16, 6000 // len(texts))
    return "\n".join(x[:per_post] for x in texts)[:6000]


def compact_document(texts, maximum=8, per_post=120):
    values = [str(x).strip() for x in texts if str(x).strip()]
    if not values:
        return "无可用文本"
    if len(values) > maximum:
        take = np.linspace(0, len(values) - 1, maximum).round().astype(int)
        values = [values[i] for i in take]
    return "；".join(x[:per_post] for x in values)


def message_for(row):
    source = compact_document(row.get("texts", []))
    return ("将以下社交媒体文字改写为一句中性的医学症状语义检索查询。"
            "只保留原文明确表达的感受、症状或状态；不要诊断、不要推断未写出的症状，"
            "不要给建议。若原文没有明确的症状或状态，请输出：未见明确症状表达。\n"
            f"社交媒体文字：{source}")


def query_texts(rows, cache: Path, model_path, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS query (idx INTEGER PRIMARY KEY, user_id TEXT UNIQUE NOT NULL, prompt_hash TEXT NOT NULL, query TEXT NOT NULL, backend TEXT NOT NULL, created_at REAL NOT NULL)")
        existing = {i: (uid, h) for i, uid, h in connection.execute("SELECT idx,user_id,prompt_hash FROM query")}
        for i, (uid, digest) in existing.items():
            if not 0 <= i < len(rows) or rows[i]["user_id"] != uid or hashlib.sha256(message_for(rows[i]).encode("utf-8")).hexdigest() != digest:
                raise ValueError("The query cache belongs to different users or prompts")
        pending = [i for i in range(len(rows)) if i not in existing]
        if pending:
            import time
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, padding_side="left")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False, torch_dtype=dtype, low_cpu_mem_usage=False).to(device).eval()
            for start in range(0, len(pending), 8):
                indices = pending[start:start+8]
                messages = [message_for(rows[i]) for i in indices]
                prompts = [tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True) for message in messages]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                with torch.inference_mode():
                    tokens = model.generate(**encoded, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                queries = tokenizer.batch_decode(tokens[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                connection.executemany("INSERT INTO query VALUES(?,?,?,?,?,?)", [(i, rows[i]["user_id"], hashlib.sha256(message.encode("utf-8")).hexdigest(), query.strip(), "Qwen2.5-3B-Instruct", time.time()) for i, message, query in zip(indices, messages, queries)])
                connection.commit()
                print(f"Queries: {len(existing) + start + len(indices)}/{len(rows)}", flush=True)
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        queries = connection.execute("SELECT idx,user_id,query FROM query ORDER BY idx").fetchall()
    if len(queries) != len(rows) or any(i != row[0] or rows[i]["user_id"] != row[1] for i, row in enumerate(queries)):
        raise ValueError("Query cache identities or order do not match the users")
    return [row[2] for row in queries]


def fit_encoder(rows, user_id, split, output: Path):
    train = np.flatnonzero(split == 0)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                                 max_df=.995, max_features=30000, sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform([make_document(rows[i]) for i in train])
    svd = TruncatedSVD(n_components=256, n_iter=7, random_state=42)
    svd.fit(matrix)
    encoder = {"vectorizer": vectorizer, "svd": svd,
               "fit_user_id_sha256": hashlib.sha256("\n".join(user_id[train]).encode()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, output)
    return encoder


def encode(encoder, texts, batch_size=2048):
    values = [encoder["svd"].transform(encoder["vectorizer"].transform(texts[i:i+batch_size])).astype(np.float32)
              for i in range(0, len(texts), batch_size)]
    return normalize(np.concatenate(values), norm="l2").astype(np.float32)


def retrieve(query, bank, k=5):
    valid = np.linalg.norm(query, axis=1) > 1e-8
    result = np.zeros((len(query), bank.shape[1]), dtype=np.float32)
    indices = np.full((len(query), k), -1, dtype=np.int16)
    if np.any(valid):
        similarity = query[valid] @ bank.T
        top = np.argpartition(similarity, -k, axis=1)[:, -k:]
        weights = np.take_along_axis(similarity, top, axis=1)
        weights = np.exp((weights - weights.max(1, keepdims=True)) / .07)
        weights /= weights.sum(1, keepdims=True)
        vectors = (bank[top] * weights[:, :, None]).sum(1)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        result[valid] = vectors.astype(np.float32)
        indices[valid] = top.astype(np.int16)
    return result, indices
