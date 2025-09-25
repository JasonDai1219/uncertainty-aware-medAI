from pathlib import Path
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_cp_results(df_cp, total_size: int, s_gpt: float, t_gpt: float, save_path: Path | None=None):
    xs = np.asarray(df_cp["alpha"], dtype=float)
    jitter = 0.002

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xs, df_cp["FN_baseline"], "--", color="black", lw=2, label="Baseline FN", zorder=1)
    axes[0].plot(xs - jitter, df_cp["FN_confident"], "o-", color="blue", mfc="white", mec="blue",
                 ms=7, mew=2, alpha=0.95, label="CP Confident FN", zorder=5)
    axes[0].plot(xs, df_cp["FN_oracle"], "s-", color="orange", ms=6, label="CP + Oracle FN", zorder=4)
    axes[0].plot(xs + jitter, df_cp["FN_gpt"], "^-", color="green", ms=6,
                 label=f"CP + GPT FN (s={s_gpt}, t={t_gpt})", zorder=3)

    overlap = np.isclose(df_cp["FN_confident"], df_cp["FN_baseline"])
    for x, y in zip(xs[overlap], np.asarray(df_cp["FN_confident"])[overlap]):
        axes[0].annotate("overlap", (x - jitter, y), xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color="blue")

    axes[0].set_xlabel("alpha"); axes[0].set_ylabel("False Negatives"); axes[0].set_title("FN vs alpha")
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(xs, df_cp["C_size"]/total_size, "o-", color="blue", label="C coverage")
    axes[1].plot(xs, df_cp["U_size"]/total_size, "o-", color="orange", label="U coverage")
    axes[1].set_xlabel("alpha"); axes[1].set_ylabel("Proportion of samples")
    axes[1].set_title("Coverage of Confident (C) vs Uncertain (U)")
    axes[1].grid(True); axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("Saved figure to:", save_path)
    plt.show()
    plt.close()


def plot_cp_gpt_measured(df_cp, total_size: int, save_path: Path | None=None):
    xs = np.asarray(df_cp["alpha"], dtype=float)
    jitter = 0.003

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xs, df_cp["FN_baseline"], "--", color="black", lw=2, label="Baseline FN", zorder=1)
    axes[0].plot(xs - jitter, df_cp["FN_confident"], "o-", color="blue", mfc="white", mec="blue",
                 ms=7, mew=2, alpha=0.95, label="CP Confident FN", zorder=5)
    axes[0].plot(xs, df_cp["FN_oracle"], "s-", color="orange", ms=6, label="CP + Oracle FN", zorder=4)
    axes[0].plot(xs + jitter, df_cp["FN_gpt_meas"], "^-", color="green", ms=6,
                 label="CP + GPT FN (measured)", zorder=3)

    overlap = np.isclose(df_cp["FN_confident"], df_cp["FN_baseline"])
    for x, y in zip(xs[overlap], np.asarray(df_cp["FN_confident"])[overlap]):
        axes[0].annotate("overlap", (x - jitter, y), xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color="blue")

    axes[0].set_title("FN vs alpha (GPT measured)")
    axes[0].set_xlabel("alpha"); axes[0].set_ylabel("False Negatives")
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(xs, df_cp["C_size"]/total_size, "o-", color="blue", label="C coverage")
    axes[1].plot(xs, df_cp["U_size"]/total_size, "o-", color="orange", label="U coverage")
    axes[1].set_title("Coverage vs alpha")
    axes[1].set_xlabel("alpha"); axes[1].set_ylabel("Proportion")
    axes[1].grid(True); axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("Saved figure to:", save_path)
    plt.show()
    plt.close()