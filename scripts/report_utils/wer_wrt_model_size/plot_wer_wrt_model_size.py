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


WHISPER_MODEL_SIZE_IN_M_PARAMETERS = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550,
    "large-v2": 1550
}


def main(datapath: str=typer.Argument(..., help="Path to CSV file containing WERs."),
         regression: bool=typer.Option(False, "--regression", "-r", help="Whether to plot a regression line."),
         log: bool=typer.Option(False, "--log", "-l", help="Whether to plot the x-axis on a log scale."),
         savename: str=typer.Option(None, "--savename", "-f", help="Filename of the saved plot (without the suffix).")):
    """
    `datapath` must be the filepath of a manually created CSV file containing the WERs of the models.
    
    Example:
    ```
    Model,WER (%),Size (M parameters)
    tiny,13.64,39
    base,8.47,74
    small,7.72,244
    medium,5.44,769
    large,,1550
    large-v2,,155
    ```
    """
    
    # Load data:
    df = pd.read_csv(datapath)
    df = df.dropna(subset=["WER (%)"])
    
    if regression:
        sns.regplot(data=df, x="Size (M parameters)", y="WER (%)", logx=log,
                    ci=None, scatter_kws={"s": 100})  # type: ignore
    else:
        sns.scatterplot(data=df, x="Size (M parameters)", y="WER (%)",
                        hue="Model", s=100)
        if log:
            plt.xscale("log")
        
    if log:
        plt.xlabel("Size (M parameters) [log]")
    
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
