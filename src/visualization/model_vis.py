# src/visualization/model_vis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


def _extract_y(y):
    if isinstance(y, pd.DataFrame):
        return y["glaucoma"].astype(int).to_numpy()
    if isinstance(y, pd.Series):
        return y.astype(int).to_numpy()
    return np.asarray(y).astype(int).ravel()


def plot_confusion_matrix(y_true, y_pred, save_path: Path | None=None, title="Confusion Matrix"):
    y_true = _extract_y(y_true)
    y_pred = _extract_y(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()


def plot_prob_distribution(y_true, proba_pos, save_path: Path | None=None, threshold: float = 0.5,
                           highlight_range: tuple[float,float]=(0.4,0.6)):
    y_true = _extract_y(y_true)
    proba_pos = np.asarray(proba_pos).ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(proba_pos[y_true == 0], bins=20, alpha=0.6, label="True 0", density=True)
    ax.hist(proba_pos[y_true == 1], bins=20, alpha=0.6, label="True 1", density=True)
    ax.axvline(threshold, linestyle="--", color="k", label=f"threshold={threshold:.2f}")

    if highlight_range:
        ax.axvspan(highlight_range[0], highlight_range[1], color="red", alpha=0.15, label="uncertain zone")

    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Density")
    ax.set_title("Probability distribution by true label")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def plot_calibration(pipeline, X_test, y_test, n_bins: int = 10, save_path: Path | None=None,
                     highlight_range: tuple[float,float]=(0.3,0.7)):
    proba = pipeline.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(_extract_y(y_test), proba, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")

    if highlight_range:
        ax.axvspan(highlight_range[0], highlight_range[1], color="red", alpha=0.15, label="region of poor calibration")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve")
    ax.legend(); ax.grid(True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def plot_logreg_coeffs(pipeline, feature_names: list[str], top_k: int = 12, save_path: Path | None=None):
    clf = None
    for _, step in pipeline.named_steps.items():
        if hasattr(step, "coef_"):
            clf = step; break
    if clf is None:
        raise ValueError("No linear classifier with coef_ found in pipeline.")

    coefs = clf.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:top_k]
    names = [feature_names[i] for i in order]
    vals  = coefs[order]

    fig, ax = plt.subplots(figsize=(6.2, 0.38*len(names)+1.6))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title("LogReg coefficients (top |value|)")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()