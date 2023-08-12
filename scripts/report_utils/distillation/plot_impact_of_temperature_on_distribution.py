import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import softmax

import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import DEFAULT_OUTPUT_DIR


sns.set_theme(context="paper", style="ticks")


DEFAULT_LIST_TEMPERATURES = [0.01, 0.5, 1, 5, 10]

def plot_impact_of_temperature_on_distribution(n_classes: int = typer.Option(4, help="Number of classes."),
                                               title: bool = typer.Option(False, "-t", "--title", help="Whether to add a title to the plot."),
                                               seed: Optional[int] = typer.Option(None, "-s", "--seed", help="Random seed.")):
    """
    Plot the impact of temperature on the distribution of the model's predictions.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Randomly generate the logits:
    logits = np.random.randn(n_classes)
    
    # Apply temperature:
    df = pd.DataFrame({"logits": logits})
    
    # Start the index from 1:
    df.index += 1
    
    for temperature in DEFAULT_LIST_TEMPERATURES:
        df[f"$\\tau = {temperature}$"] = softmax(df["logits"] / temperature)
    df = df.drop(columns=["logits"])
    
    # Plot:
    df.plot.bar(subplots=True, legend=False, layout=(1, -1), sharey=True, figsize=(8, 3))
    
    if title:
        plt.suptitle(r"Impact of the temperature $\tau$ on the distribution of the teacher's predictions $p'_{i}(\mathbf{x})$ " + f"for {n_classes} classes")
    
    # Save figure:
    filename = "temperature_impact_on_teacher_distribution"
    savepath = (DEFAULT_OUTPUT_DIR / "temperature_impact_on_distribution").with_suffix(".png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Saved plot to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(plot_impact_of_temperature_on_distribution)
