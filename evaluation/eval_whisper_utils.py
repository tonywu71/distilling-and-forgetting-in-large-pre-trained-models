from typing import Optional
from pathlib import Path
import pandas as pd
import wandb

from utils.file_io import extract_output_savepath_from_model_path


def save_wer_to_csv(wer_metrics: pd.Series,
                    pretrained_model_name_or_path: str,
                    dataset_name: str,
                    savepath: Optional[str] = None) -> None:
    """
    Save the WER metrics to a CSV file.
    """
    if savepath is None:
        savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}.csv"
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    wer_metrics.to_csv(f"{savepath}")
    print(f"WER metrics saved to `{savepath}`.")
    return


def log_wer_to_wandb(wer_metrics: pd.Series) -> None:
    """
    Log the WER metrics to W&B.
    """
    barplot = wandb.plot.bar(wandb.Table(dataframe=wer_metrics.to_frame().reset_index()),  # type: ignore
                             label=wer_metrics.index.name,  # "Dataset"
                             value=str(wer_metrics.name),  # "WER (%)"
                             title="Per dataset WER (%)")
    wandb.log({"WER (%) for dataset group": barplot})
    return


def save_edit_metrics_to_csv(df_edit_metrics: pd.DataFrame,
                             pretrained_model_name_or_path: str,
                             dataset_name: str,
                             savepath: Optional[str] = None) -> None:
    """
    Save the edit metrics to a CSV file.
    """
    savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}-edit_metrics.csv"
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    df_edit_metrics.to_csv(savepath)
    print(f"Edit metrics saved to `{savepath}`.")
    
    
def log_edit_metrics_to_wandb(df_edit_metrics: pd.DataFrame) -> None:
    """
    Log the edit metrics to W&B.
    """
    print("\n-----------------------\n")
    
    print("All edit metrics:")
    print(df_edit_metrics)
    
    print("\n-----------------------\n")
    
    df_edit_metrics_per_dataset = df_edit_metrics.T
    df_edit_metrics_per_dataset.index.name = "Metric"
    for dataset_name in df_edit_metrics_per_dataset.columns:
        barplot = wandb.plot.bar(wandb.Table(dataframe=df_edit_metrics_per_dataset[dataset_name].to_frame().reset_index()),  # type: ignore
                                    label=df_edit_metrics_per_dataset.index.name,  # "Metric"
                                    value=dataset_name,  # should be equal to `df_edit_metrics_per_dataset.name`ß
                                    title=f"String edit metrics for {dataset_name}")
        wandb.log({f"String edit metrics for {dataset_name}": barplot})
    return
