import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Tuple
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepath_ewc: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True, resolve_path=True),
         filepath_default: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True, resolve_path=True),
         is_relative: bool = typer.Option(False, help="Whether to plot relative WER difference instead of absolute WER."),
         figsize: Tuple[float, float] = typer.Option((6, 4), help="Figure size (width, height).")):
    """
    Script that takes a CSV output from `compare_multiple_models_to_csv.py` and saves
    a plot of the evolution of WER with respect to fine-tuning checkpoints on the
    FAB dataset.
    """

    plt.figure(figsize=figsize)

    for filepath, label_suffix, ls, lw, ms in zip([filepath_ewc, filepath_default], ["EWC", "Default"], ["-", "--"], [2, 0.5], [8, 6]):
        df = pd.read_csv(filepath).set_index("steps")
        
        if is_relative:
            df = 100 * (df - df.iloc[0]) / df.iloc[0]

        l_colors = list(sns.color_palette(n_colors=len(df.columns)).as_hex())
        
        for col, color in zip(df.columns, l_colors):
            sns.lineplot(data=df, x=df.index, y=col, label=f"{col} - {label_suffix}",
                         marker="o", dashes=False, ls=ls, linewidth=lw, markersize=ms, color=color)
        
    if is_relative:
        plt.axhline(0, color="black", linestyle="--")
    
    plt.legend(loc="best")
    plt.xlabel("Steps")
    plt.ylabel("WER (%)") if not is_relative else plt.ylabel("Relative WER difference (%)")
    
    # Save figure:
    filename_stem = "wer_wrt_steps_on_fab-ewc"
    if is_relative:
        filename_stem += "-relative"
    savepath = (DEFAULT_OUTPUT_DIR / filename_stem).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
