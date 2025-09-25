from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

def _metrics_from_scores(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    proba_pos = np.asarray(proba_pos).ravel()
    y_hat = (proba_pos >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_hat)
    auroc = roc_auc_score(y_true, proba_pos) if len(np.unique(y_true)) == 2 else np.nan
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    return {
        "ACC": acc,
        "AUROC": auroc,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "RECALL": recall, "PRECISION": precision, "F1": f1,
    }

def bootstrap_baseline(y_true: np.ndarray, proba_pos: np.ndarray, n_boot: int = 500, threshold: float = 0.5, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows.append(_metrics_from_scores(y_true[idx], proba_pos[idx], threshold))

    df = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "mean": df.mean(),
        "ci_low": df.quantile(0.025),
        "ci_high": df.quantile(0.975),
    })
    return df, summary