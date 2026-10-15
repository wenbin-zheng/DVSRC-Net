"""Binary classification and ranking metrics."""

import numpy as np
from sklearn.metrics import (average_precision_score, auc, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def score(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = probability >= threshold
    precision, recall, _ = precision_recall_curve(y, probability)
    return {"precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "auroc": float(roc_auc_score(y, probability)),
            "ap": float(average_precision_score(y, probability)),
            "pr_auc_trapezoid": float(auc(recall, precision)),
            "f1": float(f1_score(y, prediction, zero_division=0)), "threshold": float(threshold)}


def summarize(rows):
    keys = ("precision", "recall", "f1", "auroc", "ap", "pr_auc_trapezoid")
    return {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "sample_sd": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None}
        for key in keys
    }
