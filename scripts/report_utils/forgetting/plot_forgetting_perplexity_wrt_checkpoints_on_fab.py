import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepath: str, is_relative: bool=False):
    """
    Script that takes a CSV output from `compare_multiple_models_to_csv.py` and saves
    a plot of the evolution of perplexity with respect to fine-tuning checkpoints on the
    FAB dataset.
    """
    
    df = pd.read_csv(filepath).set_index("Dataset")
    df.index.name = "FAB dataset"
    df = df.T
    df.index = df.index.str.extract(r'checkpoint-(\d+)-implicit_lm-perplexity-fab').astype(int).values.flatten()  # type: ignore
    
    if is_relative:
        df = 100 * (df - df.iloc[0]) / df.iloc[0]
    
    plt.figure(figsize=(6, 4))
    sns.lineplot(df.iloc[:, :-1], marker="o", dashes=False)
    if is_relative:
        plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Perplexity") if not is_relative else plt.ylabel("Relative perplexity difference (%)")
    
    # Save figure:
    if is_relative:
        savepath = (DEFAULT_OUTPUT_DIR / "report" / "compare_perplexity_multiple_models" / (Path(filepath).stem + "-relative")).with_suffix(".png")
    else:
        savepath = (DEFAULT_OUTPUT_DIR / "report" / "compare_perplexity_multiple_models" / Path(filepath).stem).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
