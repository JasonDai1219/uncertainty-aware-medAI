import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.data.preprocess import split_X_y

def calibrate_threshold(pipe, calib_df: pd.DataFrame, alpha: float) -> float:
    Xc, yc = split_X_y(calib_df)
    proba = pipe.predict_proba(Xc)
    p_true = proba[np.arange(len(yc)), yc]
    A = 1.0 - p_true
    q = np.quantile(A, 1.0 - alpha, method="higher")
    return float(q)

def cp_partition(pipe, df: pd.DataFrame, q: float) -> Dict[str, np.ndarray]:
    X, y = split_X_y(df)
    proba = pipe.predict_proba(X)
    yhat = proba.argmax(axis=1)
    maxp = proba.max(axis=1)
    conf_mask = (1.0 - maxp) <= q
    region = np.where(conf_mask, "C", "U")
    return {
        "y_true": y,
        "y_pred": yhat,
        "proba": proba,
        "region": region
    }

def summarize_counts(y_true, y_pred, region) -> Dict[str, int]:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); region = np.asarray(region)
    C = region == "C"; U = ~C
    TP_c = np.sum((y_true==1) & (y_pred==1) & C)
    FN_c = np.sum((y_true==1) & (y_pred==0) & C)
    FP_c = np.sum((y_true==0) & (y_pred==1) & C)
    TN_c = np.sum((y_true==0) & (y_pred==0) & C)

    U_pos = np.sum((y_true==1) & U)
    U_neg = np.sum((y_true==0) & U)

    return {
        "TP_confident": int(TP_c),
        "FN_confident": int(FN_c),
        "FP_confident": int(FP_c),
        "TN_confident": int(TN_c),
        "U_pos": int(U_pos),
        "U_neg": int(U_neg),
        "C_size": int(C.sum()),
        "U_size": int(U.sum())
    }