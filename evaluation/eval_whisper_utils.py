from typing import Optional, List
from pathlib import Path
import json
import pandas as pd
import wandb

from utils.file_io import extract_output_savepath_from_model_path


def save_edit_metrics_to_csv(df_edit_metrics: pd.DataFrame,
                             pretrained_model_name_or_path: str,
                             dataset_name: str,
                             savepath: Optional[str] = None,
                             suffix: Optional[str] = None) -> None:
    """
    Save the edit metrics to a CSV file.
    """
    if suffix:
        savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}-{suffix}.csv"
    else:
        savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}.csv"
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    df_edit_metrics.to_csv(savepath)
    print(f"Edit metrics saved to `{savepath}`.")
    return


def log_wer_to_wandb(wer_metrics: pd.Series,
                     suffix: Optional[str] = None) -> None:
    """
    Log the WER metrics to W&B.
    """
    plot_title = f"Per dataset WER (%) - {suffix}" if suffix else "Per dataset WER (%)"
    barplot = wandb.plot.bar(wandb.Table(dataframe=wer_metrics.to_frame().reset_index()),  # type: ignore
                             label=wer_metrics.index.name,  # "Dataset"
                             value=str(wer_metrics.name),  # "WER (%)"
                             title=plot_title)
    log_title = f"WER (%) for dataset group - {suffix}" if suffix else "WER (%) for dataset group"
    wandb.log({log_title: barplot})
    return
    
    
def log_edit_metrics_to_wandb(df_edit_metrics: pd.DataFrame,
                              suffix: Optional[str] = None) -> None:
    """
    Log the edit metrics to W&B.
    """
    df_edit_metrics_per_dataset = df_edit_metrics.T
    df_edit_metrics_per_dataset.index.name = "Metric"
    for dataset_name in df_edit_metrics_per_dataset.columns:
        title = f"String edit metrics for {dataset_name} - {suffix}" if suffix else f"String edit metrics for {dataset_name}"
        barplot = wandb.plot.bar(wandb.Table(dataframe=df_edit_metrics_per_dataset[dataset_name].to_frame().reset_index()),  # type: ignore
                                 label=df_edit_metrics_per_dataset.index.name,  # "Metric"
                                 value=dataset_name,  # should be equal to `df_edit_metrics_per_dataset.name`ß
                                 title=title)
        wandb.log({title: barplot})
    return


def save_preds_to_json(references: List[str],
                       predictions: List[str],
                       savepath: str) -> None:
    """
    Export `references` and `predictions` to a JSON file.
    """
    data = {'references': references, 'predictions': predictions}
    with open(savepath, 'w') as file:
        json.dump(data, file)
    return
