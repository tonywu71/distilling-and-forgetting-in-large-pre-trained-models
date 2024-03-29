import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(k: int = typer.Argument(..., help="Maximum value of k."),
         title: bool = typer.Option(False, "-t", "--title", help="Whether to add a title to the plot."),
         figsize: Tuple[int, int] = typer.Option((6, 2), "-s", "--figsize", help="Figure size.")):
    LIST_BETA = [0.1, 1, 2, 5]
    assert k >= 1, "`k` must be greater than or equal to 1."
    
    # Create dataframe:
    df = pd.DataFrame({"k": np.arange(1, k + 1)})
    
    # Compute the weights:
    for beta in LIST_BETA:
        df[beta] = np.exp(- beta * (df["k"] - 1))
    df = df.set_index("k")
    df = df / df.sum()  # normalize the weights
    
    df = df.add_prefix(r"$\beta = ")
    df = df.add_suffix(r"$")
    
    # Plot:
    df.plot.bar(subplots=True, legend=False, layout=(1, -1), sharey=True, figsize=figsize)
    if title:
        plt.suptitle(r"Distribution of $w_k$ for $\beta \in \{0.1, 1, 2, 5\}$")
    
    # Save figure:
    savepath = (DEFAULT_OUTPUT_DIR / f"weight_seq_level_ranked_k_{k}").with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
