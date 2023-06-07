import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datetime import datetime

from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


def main(filepath: str = typer.Argument(..., help="Path to the CSV file."),
         x: str = typer.Option(..., "--x", "-x", help="Column name for x. Should be a hyperparameter."),
         y: str = typer.Option(..., "--y", "-y", help="Column name for y. Should be a metric."),
         r2_loc: str = typer.Option("upper right", "--r2-loc", "-r2", help="Location of the R2 score annotation."),
         filename: str = typer.Option(None, "--filename", "-f", help="Filename of the plot (without the suffix).")):
    
    # Load data:
    df = pd.read_csv(filepath, dtype={"learning_rate": str}, index_col="idx")
    
    # Preprocess data:
    df.index = df.index.str.split("_").str[0].astype(int)
    df.index.name = "Order"
    
    # Compute the R2 score using scipy.stats:
    r, p = pearsonr(df[x], df[y])
    
    # Plot:
    sns.lmplot(data=df, x=x, y=y)
    
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
    if filename is None:
        filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    savepath = (DEFAULT_OUTPUT_DIR / "report" / "hpt" / filename).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
