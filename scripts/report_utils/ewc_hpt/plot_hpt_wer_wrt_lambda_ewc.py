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


def main(dirpath: Path = typer.Argument(..., file_okay=False, dir_okay=True),
         figsize: Tuple[float, float] = typer.Option((12, 3), help="Figure size (width, height).")):
    assert dirpath.is_dir(), f"{dirpath} is not a directory"
    
    # Print the number of files found:
    list_filepaths = sorted(list(dirpath.glob("*")))
    print(f"Found {len(list_filepaths)} files.")
    list_df = [pd.read_csv(filepath).assign(lambda_ewc=filepath.stem) for filepath in dirpath.glob("*")]
    
    # Load and concatenate all CSV files:
    list_df = [pd.read_csv(filepath).assign(lambda_ewc=filepath.stem) for filepath in dirpath.glob("*")]
    df = pd.concat(list_df).reset_index(drop=True)
    df["lambda_ewc"] = df["lambda_ewc"].str.extract(r'lambda_(.+)').astype(float)
    df = df[df["lambda_ewc"]!= 0]  # drop rows where lambda = 0
    
    # Plot:
    n_cols = df["Dataset"].nunique()
    fig, axis = plt.subplots(1, n_cols, figsize=figsize)

    for dataset_name, ax in zip(df["Dataset"].unique(), axis.ravel()):
        df_curr = df.loc[df["Dataset"]==dataset_name]
        sns.barplot(data=df_curr, y="lambda_ewc", x="WER (%)", orient="h", ax=ax)
        if float("inf") in df_curr["lambda_ewc"].unique():
            x_vanilla = df_curr.loc[df_curr["lambda_ewc"] == float("inf"), "WER (%)"].item()
            ax.axvline(x=x_vanilla, label=f"Original model (WER = {x_vanilla}%)", c="black", ls="--")
            ax.legend(loc="lower left")
        ax.set_title(dataset_name)
    
    # Save the DataFrame to CSV:
    savepath = DEFAULT_OUTPUT_DIR / "ewc_hpt-wer_wrt_lambda.png"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath)
    
    print(f"Figure saved to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
