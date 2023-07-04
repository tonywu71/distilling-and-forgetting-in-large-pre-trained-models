import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Tuple

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR

sns.set_theme(context="paper", style="ticks")


def main(datapath: str = typer.Argument(..., help="Path to CSV file containing WERs."),
         log: bool = typer.Option(False, "--log", "-l", help="Whether to plot the x-axis on a log scale."),
         plot_expected: bool = typer.Option(False, "--plot-expected", "-i", help="Whether to plot the expected distilled model."),
         figsize: Tuple[float, float] = typer.Option((7, 3), help="Figure size (width, height).")):
    """
    `datapath` must be the filepath of a manually created CSV file containing the WER and the model size for each model.
    
    Example:
    ```
    Model,WER (%),Size (M parameters)
    tiny,13.64,39
    base,8.47,74
    small,7.72,244
    medium,5.44,769
    large,,1550
    large-v2,,1550
    ```
    """
    
    # Load data:
    df = pd.read_csv(datapath)
    df = df.dropna(subset=["WER (%)"])
    
    # Prepare markers:
    markers = {model: "o" for model in df["Model"].unique()}
    
    # Add expected distilled model:
    if plot_expected:
        size_expected = df["Size (M parameters)"].min()
        wer_expected = df["WER (%)"].min()
        row_expected = pd.DataFrame.from_dict({
            "Model": ["Expected distilled model"],
            "WER (%)": [wer_expected],
            "Size (M parameters)": [size_expected]
        })
        df = pd.concat([df, row_expected])  # type: ignore
        markers["Expected distilled model"] = "^"
    
    # Plot:
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(data=df, x="Size (M parameters)", y="WER (%)", hue="Model",
                    s=400, style="Model", markers=markers, ax=ax)  # type: ignore
    if log:
        plt.xscale("log")
        
    if log:
        plt.xlabel("Size (M parameters) [log]")
    
    if plot_expected:
        plt.axvline(x=size_expected, color="black", linestyle="dashed")
        plt.axhline(y=wer_expected, color="black", linestyle="dashed")
    
    plt.ylim(df["WER (%)"].min() - 1, df["WER (%)"].max() + 1)
    
    # Save figure:
    savepath = (DEFAULT_OUTPUT_DIR / "plot_wer_wrt_model_size").with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
