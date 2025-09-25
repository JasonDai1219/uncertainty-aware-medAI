from pathlib import Path
import matplotlib.pyplot as plt

def plot_cp_results(df_cp, total_size: int, s_gpt: float, t_gpt: float, save_path: Path | None=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_cp["alpha"], df_cp["FN_baseline"], "--", label="Baseline FN", color="black")
    axes[0].plot(df_cp["alpha"], df_cp["FN_confident"], "o-", label="CP Confident FN")
    axes[0].plot(df_cp["alpha"], df_cp["FN_oracle"], "o-", label="CP + Oracle FN")
    axes[0].plot(df_cp["alpha"], df_cp["FN_gpt"], "o-", label=f"CP + GPT FN (s={s_gpt}, t={t_gpt})")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("False Negatives")
    axes[0].set_title("FN vs alpha")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df_cp["alpha"], df_cp["C_size"]/total_size, "o-", label="C coverage")
    axes[1].plot(df_cp["alpha"], df_cp["U_size"]/total_size, "o-", label="U coverage")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Proportion of samples")
    axes[1].set_title("Coverage of Confident (C) vs Uncertain (U)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def plot_cp_gpt_measured(df_cp, total_size: int, save_path: Path | None=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_cp["alpha"], df_cp["FN_baseline"], "--", color="black", label="Baseline FN")
    axes[0].plot(df_cp["alpha"], df_cp["FN_confident"], "o-", label="CP Confident FN")
    axes[0].plot(df_cp["alpha"], df_cp["FN_oracle"], "o-", label="CP + Oracle FN")
    axes[0].plot(df_cp["alpha"], df_cp["FN_gpt_meas"], "o-", label="CP + GPT FN (measured)")

    axes[0].set_title("FN vs alpha (GPT measured)")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("False Negatives")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df_cp["alpha"], df_cp["C_size"]/total_size, "o-", label="C coverage")
    axes[1].plot(df_cp["alpha"], df_cp["U_size"]/total_size, "o-", label="U coverage")
    axes[1].set_title("Coverage vs alpha")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Proportion")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("Saved figure to:", save_path)
    plt.show()
    plt.close()