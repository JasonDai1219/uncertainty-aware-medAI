import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from src.data.preprocess import build_preprocessor, split_X_y

def train_logreg(train_df: pd.DataFrame):
    X_tr, y_tr = split_X_y(train_df)
    preproc = build_preprocessor()
    clf = LogisticRegression(max_iter=200, class_weight=None)
    pipe = Pipeline([("prep", preproc), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    return pipe

def evaluate(pipe, df: pd.DataFrame, threshold=0.5):
    X, y = split_X_y(df)
    proba = pipe.predict_proba(X)[:, 1]
    yhat = (proba >= threshold).astype(int)

    cm = confusion_matrix(y, yhat, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    acc = (TP+TN)/cm.sum()
    auroc = roc_auc_score(y, proba)

    metrics = {
        "ACC": acc,
        "AUROC": auroc,
        "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
        "RECALL": TP / (TP + FN + 1e-12),
        "PRECISION": TP / (TP + FP + 1e-12),
        "F1": (2*TP) / (2*TP + FP + FN + 1e-12)
    }
    return metrics, proba, yhat