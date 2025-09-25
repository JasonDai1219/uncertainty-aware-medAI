from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONT_COLS = ["age", "ocular_pressure", "MD", "PSD", "cornea_thickness", "RNFL4.mean"]
CAT_COLS  = ["RL", "GHT"]
LABEL_COL = "glaucoma"

def _ensure_dir(p: Path | str | None):
    if p is None: return
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def plot_label_distribution(df: pd.DataFrame, save_path: Path | None = None):
    counts = df[LABEL_COL].value_counts().sort_index()
    props  = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(counts.index.astype(str), counts.values)
    for i,v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({props.iloc[i]:.2f})", ha="center", va="bottom")
    ax.set_title("Label distribution (glaucoma)")
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path: _ensure_dir(save_path); plt.savefig(save_path, dpi=180)
    else: plt.show()
    plt.close()

def plot_continuous_hists_by_label(df: pd.DataFrame, cols=CONT_COLS, bins=20, save_path: Path | None=None):
    pos = df[df[LABEL_COL]==1]; neg = df[df[LABEL_COL]==0]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.6*n))
    if n==1: axes=[axes]
    for ax, col in zip(axes, cols):
        ax.hist(neg[col], bins=bins, alpha=0.6, label="0", density=True)
        ax.hist(pos[col], bins=bins, alpha=0.6, label="1", density=True)
        ax.set_title(f"{col} by label"); ax.set_ylabel("Density"); ax.legend(title="glaucoma")
    axes[-1].set_xlabel("Value")
    plt.tight_layout()
    if save_path: _ensure_dir(save_path); plt.savefig(save_path, dpi=180)
    else: plt.show()
    plt.close()

def plot_violin_by_label(df: pd.DataFrame, cols=("MD","PSD","RNFL4.mean"), save_path: Path | None=None):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4))
    if n==1: axes=[axes]
    for ax, col in zip(axes, cols):
        data = [df[df[LABEL_COL]==0][col].values, df[df[LABEL_COL]==1][col].values]
        ax.boxplot(data, labels=["0","1"], showfliers=True)
        ax.set_title(f"{col} by label"); ax.set_xlabel("glaucoma"); ax.set_ylabel(col)
    plt.tight_layout()
    if save_path: _ensure_dir(save_path); plt.savefig(save_path, dpi=180)
    else: plt.show()
    plt.close()

def plot_categorical_counts_by_label(df: pd.DataFrame, cols=CAT_COLS, save_path: Path | None=None):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4))
    if n==1: axes=[axes]
    for ax, col in zip(axes, cols):
        ctab = pd.crosstab(df[col], df[LABEL_COL]).sort_index()
        x = np.arange(len(ctab.index))
        width = 0.38
        ax.bar(x - width/2, ctab.get(0,0), width, label="0")
        ax.bar(x + width/2, ctab.get(1,0), width, label="1")
        ax.set_xticks(x); ax.set_xticklabels(ctab.index.astype(str))
        ax.set_title(f"{col} vs label"); ax.set_xlabel(col); ax.set_ylabel("Count"); ax.legend(title="glaucoma")
    plt.tight_layout()
    if save_path: _ensure_dir(save_path); plt.savefig(save_path, dpi=180)
    else: plt.show()
    plt.close()

def plot_corr_heatmap(df: pd.DataFrame, cols=CONT_COLS, save_path: Path | None=None):
    M = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.8,5.2))
    im = ax.imshow(M.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Spearman correlation (continuous features)")
    plt.tight_layout()
    if save_path: _ensure_dir(save_path); plt.savefig(save_path, dpi=180)
    else: plt.show()
    plt.close()