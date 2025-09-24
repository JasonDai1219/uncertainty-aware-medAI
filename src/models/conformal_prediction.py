import numpy as np
import pandas as pd

def calibrate_threshold(pipe, calib_df, alpha: float):
    Xc, yc = split_X_y(calib_df)
    p = pipe.predict_proba(Xc)
    p_true = p[np.arange(len(yc)), yc]
    A = 1.0 - p_true
    return float(np.quantile(A, 1.0 - alpha, method="higher"))

def cp_partition(pipe, df, q: float):
    X, y = split_X_y(df)
    p = pipe.predict_proba(X)
    yhat = p.argmax(axis=1)
    maxp = p.max(axis=1)
    region = np.where((1.0 - maxp) <= q, "C", "U")
    return y, yhat, region

def summarize(y, yhat, region):
    y = np.asarray(y); yhat = np.asarray(yhat); region = np.asarray(region)
    C = region == "C"; U = ~C
    TPc = np.sum((y==1)&(yhat==1)&C)
    FNc = np.sum((y==1)&(yhat==0)&C)
    FPc = np.sum((y==0)&(yhat==1)&C)
    TNc = np.sum((y==0)&(yhat==0)&C)
    U_pos = np.sum((y==1)&U)
    U_neg = np.sum((y==0)&U)
    return dict(
        TP_confident=int(TPc), FN_confident=int(FNc),
        FP_confident=int(FPc),