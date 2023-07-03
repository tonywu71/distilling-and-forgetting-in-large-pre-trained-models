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
         figsize: Tuple[float, float] = typer.Option((6, 3), "--figsize", "-f", help="Figure size."),):
    """
    """
    assert dirpath.is_dir(), f"{dirpath} is not a directory"
    
    # Print the number of files found:
    list_filepaths = sorted(list(dirpath.glob("*")))
    print(f"Found {len(list_filepaths)} files.")
    
    # Load and concatenate all CSV files:
    list_df = [pd.read_csv(filepath).assign(k=filepath.stem) for filepath in list_filepaths]
    df = pd.concat(list_df).reset_index(drop=True)
    df["k"] = df["k"].str.extract(r'[\w+]-k_(\d+)')
    df = df.set_index("k").sort_index()
    
    # Print the DataFrame:
    print("Result:")
    print(df)
    print()
    
    # Plot:
    fig, ax = plt.subplots(figsize=figsize)
    df.plot.bar(ax=ax)
        
    # Save the DataFrame to CSV:
    savepath = DEFAULT_OUTPUT_DIR / "wer_wrt_beam_size.png"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath)
    
    print(f"Figure saved to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
