import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepath: str = typer.Argument(..., help="Path to the CSV file."),
         r2_loc: str = typer.Option("upper left", "--r2-loc", "-r2", help="Location of the R2 score annotation.")):
    
    # Load data:
    df = pd.read_csv(filepath)
    x = "Size (M parameters)"
    y = "Average latency (ms)"
    
    # Compute the R2 score using scipy.stats:
    r, p = pearsonr(df[x], df[y])
    
    # Plot:
    sns.lmplot(data=df, x=x, y=y, ci=None)
    
    # Annotate the plot with the R2 score:
    if r2_loc == "upper left":
        plt.annotate(f"$R^2$ = {r**2:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", size=14)
    elif r2_loc == "upper right":
        plt.annotate(f"$R^2$ = {r**2:.2f}", xy=(0.95, 0.95), xycoords="axes fraction", size=14, horizontalalignment="right")
    elif r2_loc == "bottom left":
        plt.annotate(f"$R^2$ = {r**2:.2f}", xy=(0.05, 0.05), xycoords="axes fraction", size=14)
    elif r2_loc == "bottom right":
        plt.annotate(f"$R^2$ = {r**2:.2f}", xy=(0.95, 0.05), xycoords="axes fraction", size=14, horizontalalignment="right")
    else:
        pass
    
    # Save figure:    
    savepath = (DEFAULT_OUTPUT_DIR / "plot_size_wrt_speed").with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
