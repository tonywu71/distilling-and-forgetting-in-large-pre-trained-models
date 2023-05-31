from torch import le
import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List
from datetime import datetime

from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepaths: List[str],
         power_law: bool=typer.Option(False, "--power-law", "-p", help="Whether to use a power law for the regression (i.e. log-log)."),
         xlim: List[float]=typer.Option(None, "--xlim", "-x", help="Limits of the x-axis."),
         ylim: List[float]=typer.Option(None, "--ylim", "-y", help="Limits of the y-axis."),
         filename: str=typer.Option(None, "--filename", "-f", help="Filename of the plot (without the suffix).")):
    """
    Script that takes a list of CSV outputs from `merge_wer_and_ppl_to_csv.py` and saves
    a scatterplot of word error rate (WER) with respect to perplexity (PPL).
    """
    
    
    # Load data:
    list_df = [pd.read_csv(filepath, index_col=0) for filepath in filepaths]
    df = pd.concat(list_df, axis=0)
    
    if power_law:
        df["Perplexity"] = df["Perplexity"].apply(lambda x: np.log(x))
        df["WER (%)"] = df["WER (%)"].apply(lambda x: np.log(x))
    
    
    # Compute the R2 score using scipy.stats:
    r, p = pearsonr(df["Perplexity"], df["WER (%)"])
    
    
    # Plot:
    g = sns.lmplot(df, x="Perplexity", y="WER (%)", hue="Model", fit_reg=False, facet_kws=dict(legend_out=False))
    sns.regplot(df, x="Perplexity", y="WER (%)", scatter=False,
                line_kws=dict(color="black", linestyle="--"),
                ax=g.axes[0, 0])  # type: ignore
    
    
    # Annotate the plot with the R2 score:
    plt.annotate(f"$R^2$ = {r**2:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", size=14)
    
    
    if power_law:
        plt.xlabel("Log perplexity")
        plt.ylabel("Log WER (%)")
    
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Save figure:
    if filename is None:
        filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    savepath = (DEFAULT_OUTPUT_DIR / "report" / "plot_wer_wrt_perplexity" / filename).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
