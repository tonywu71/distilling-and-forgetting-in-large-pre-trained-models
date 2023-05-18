import re
import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List
from datetime import datetime

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepaths: List[str],
         regression: bool=typer.Option(False, "--regression", "-r", help="Whether to plot a regression line."),
         filename: str=typer.Option(None, "--filename", "-f", help="Filename of the plot (without the suffix).")):
    """
    Script that takes a list of CSV outputs from `merge_wer_and_ppl_to_csv.py` and saves
    a scatterplot of word error rate (WER) with respect to perplexity (PPL).
    """
    list_df = [pd.read_csv(filepath, index_col=0) for filepath in filepaths]  # TODO: check
    
    df = pd.concat(list_df, axis=0)
    
    # Plot:
    if not regression:
        sns.scatterplot(df, x="WER (%)", y="Perplexity", hue="Model")
    else:
        sns.lmplot(df, x="WER (%)", y="Perplexity", hue="Model", facet_kws=dict(legend_out=False))
    
    # Save figure:
    if filename is None:
        filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if regression:
        filename += "-regression"
    savepath = (DEFAULT_OUTPUT_DIR / "report" / "plot_wer_wrt_perplexity" / filename).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
