import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


# --- Kept for reference ---
# WHISPER_MODEL_SIZE_IN_M_PARAMETERS = {
#     "tiny": 39,
#     "base": 74,
#     "small": 244,
#     "medium": 769,
#     "large": 1550,
#     "large-v2": 1550
# }


def main(datapath: str=typer.Argument(..., help="Path to CSV file containing WERs."),
         regression: bool=typer.Option(False, "--regression", "-r", help="Whether to plot a regression line."),
         log: bool=typer.Option(False, "--log", "-l", help="Whether to plot the x-axis on a log scale."),
         plot_expected: bool=typer.Option(False, "--plot-expected", "-i", help="Whether to plot the expected distilled model."),
         savename: str=typer.Option(None, "--savename", "-f", help="Filename of the saved plot (without the suffix).")):
    """
    `datapath` must be the filepath of a manually created CSV file containing the WER and the model size for each model.
    If there is a "distilled" model, an arrow from the smallest model to "distilled" will be drawn.
    
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
    df = df.dropna(subset=["WER on LibriSpeech clean (%)"])
    
    markers = {model: "o" for model in df["Model"].unique()}
    
    if plot_expected:
        size_expected = df["Size (M parameters)"].min()
        wer_expected = df["WER on LibriSpeech clean (%)"].min()
        row_expected = pd.DataFrame.from_dict({
            "Model": ["Expected distilled model"],
            "WER on LibriSpeech clean (%)": [wer_expected],
            "Size (M parameters)": [size_expected]
        })
        df = pd.concat([df, row_expected])  # type: ignore
        markers["Expected distilled model"] = "^"
    
    fig, ax = plt.subplots(figsize=(7, 3))  # for the poster
    # fig, ax = plt.subplots(figsize=(8, 6))
    
    if regression:
        sns.regplot(data=df, x="Size (M parameters)", y="WER on LibriSpeech clean (%)",
                    logx=log, ci=None, scatter_kws={"s": 400},
                    ax=ax)
    else:
        sns.scatterplot(data=df, x="Size (M parameters)", y="WER on LibriSpeech clean (%)", hue="Model",
                        s=400, style="Model", markers=markers, ax=ax)  # type: ignore
        if log:
            plt.xscale("log")
        
    if log:
        plt.xlabel("Size (M parameters) [log]")
    
    
    if plot_expected:
        plt.axvline(x=size_expected, color="black", linestyle="dashed")
        plt.axhline(y=wer_expected, color="black", linestyle="dashed")
    
    
    if "distilled" in df["Model"].unique():
        # Draw arrow from "tiny" to "distilled":
        tiny = df[df["Model"] == "tiny"]
        distilled = df[df["Model"] == "distilled"]
        plt.annotate("", xy=(distilled["Size (M parameters)"], distilled["WER on LibriSpeech clean (%)"]),
                     xytext=(tiny["Size (M parameters)"], tiny["WER on LibriSpeech clean (%)"]),
                     arrowprops=dict(arrowstyle="->", color="black", lw=3))
        # Add text to the right of the center of the arrow:
        # plt.text(x=distilled["Size (M parameters)"] + 2., y=distilled["WER on LibriSpeech clean (%)"] + 0.5, s="Distillation", fontsize=14)
        
        plt.text(x=tiny["Size (M parameters)"] + 4.,
                 y=(tiny["WER on LibriSpeech clean (%)"].values + distilled["WER on LibriSpeech clean (%)"].values) / 2,
                 s="Distillation",
                 fontsize=12)
    
    # Save figure:
    if savename is None:
        savename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    savepath = (DEFAULT_OUTPUT_DIR / "report" / "wer_wrt_model_size" / savename).with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
