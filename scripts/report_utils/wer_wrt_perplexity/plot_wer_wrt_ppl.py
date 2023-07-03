import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepaths: List[str],
         kind: str=typer.Option(None, "--kind", "-k",
                                help="Kind of plot to generate. Must be one of `None`, `regression`, or `jointplot`."),
         logx: bool=typer.Option(False, "--logx", help="Whether to use a log scale for the x-axis."),
         logy: bool=typer.Option(False, "--logy", help="Whether to use a log scale for the y-axis."),
         xlim: List[float]=typer.Option(None, "--xlim", "-x", help="Limits of the x-axis."),
         ylim: List[float]=typer.Option(None, "--ylim", "-y", help="Limits of the y-axis.")):
    """
    Script that takes a list of CSV outputs from `merge_wer_and_ppl_to_csv.py` and saves
    a scatterplot of word error rate (WER) with respect to perplexity (PPL).
    """
    
    # Load data:
    list_df = [pd.read_csv(filepath, index_col=0) for filepath in filepaths]
    df = pd.concat(list_df, axis=0)
    
    if logx:
        df["Perplexity"] = df["Perplexity"].apply(lambda x: np.log(x))
    if logy:
        df["WER (%)"] = df["WER (%)"].apply(lambda x: np.log(x))
    
    # Plot:
    if kind is None:
        sns.scatterplot(df, x="Perplexity", y="WER (%)", hue="Model")
    elif kind == "regression":
        sns.lmplot(df, x="Perplexity", y="WER (%)", hue="Model", facet_kws=dict(legend_out=False))
    else:
        raise ValueError(f"Invalid `kind` value: `{kind}`. Must be one of `None`, `regression`, or `jointplot`.")
    
    if logx:
        plt.xlabel("Log perplexity")
    if logy:
        plt.ylabel("Log WER (%)")
    
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Save figure:
    savepath = (DEFAULT_OUTPUT_DIR / "plot_wer_wrt_perplexity").with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
