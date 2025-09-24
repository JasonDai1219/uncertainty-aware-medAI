from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from math import sqrt
from dataclasses import dataclass

def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

def sensitivity_specificity(cm: Dict[str, int]) -> Tuple[float, float]:
    TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    s = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    t = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    return float(s), float(t)

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion (95% 默认)"""
    if n == 0 or np.isnan(p):
        return (np.nan, np.nan)
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    margin = (z / denom) * sqrt((p*(1-p)/n) + (z**2)/(4*n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))

def new_errors(FN_confident, FP_confident, U_pos, U_neg, s, t):
    FN_new = FN_confident + (1.0 - s) * U_pos
    FP_new = FP_confident + (1.0 - t) * U_neg
    return float(FN_new), float(FP_new)