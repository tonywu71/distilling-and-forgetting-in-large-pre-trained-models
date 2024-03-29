import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Tuple

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepath: str = typer.Argument(..., help="Path to CSV file that contains all eval results for TAC and for different values of gamma."),
         is_relative: bool = typer.Option(False, help="Whether to plot relative forgetting w.r.t. the first step (i.e., 0% forgetting)."),
         figsize: Tuple[float, float] = typer.Option((6, 4), help="Figure size (width, height).")):
    df = pd.read_csv(filepath)
    
    if is_relative:
        list_df_gamma = []
        for gamma in df["gamma"].unique():
            df_gamma = df.loc[df["gamma"]==gamma]
            df_gamma.iloc[:, 1:-1] = 100 * (df_gamma - df_gamma.iloc[0, 1:-1]) / df_gamma.iloc[0, 1:-1]
            list_df_gamma.append(df_gamma)
        df =  pd.concat(list_df_gamma, axis=0)
    
    df["gamma"] = df["gamma"].astype("category")
    
    for col_language in df.columns[1:-1]:
        plt.figure(figsize=figsize)
        sns.lineplot(data=df, x="steps", y=col_language, hue="gamma", marker="o", dashes=False)
        if is_relative:
            plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Steps")
        plt.ylabel(col_language) if not is_relative else plt.ylabel(f"Relative Δ{col_language}")
    
        # Save figure:
        filename_stem = "plot_tac_forgetting"
        if is_relative:
            filename_stem += "_relative"
        savepath = (DEFAULT_OUTPUT_DIR / filename_stem).with_suffix(".png")
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(savepath)
        print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
