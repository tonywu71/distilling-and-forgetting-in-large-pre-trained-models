import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from typing import Optional
from pathlib import Path

import pandas as pd

from utils.utils_compare_wer import post_process_esb_librispeech, post_process_esb_ami, post_process_mls


def main(filepath_distilled: str=typer.Argument(..., help="Path to CSV file."),
         filepath_small_model: str=typer.Option(..., help="Path to CSV file for the small model."),
         filepath_large_model: str=typer.Option(..., help="Path to CSV file for the large model."),
         dataset_group: Optional[str]=typer.Option(None, help="Dataset group to use. Either 'esb_librispeech', 'esb_ami', or 'mls'.")):
    """
    Script that takes 2 CSV outputs from `eval_whisper_on_esb.py` and outputs a comparison table in LaTeX.
    Use the `dataset_group` option to specify the dataset group to use for additional statistics.ß
    
    To be used for LaTeX table generation in reports.
    """
    
    # Read CSV files:
    df_distilled = pd.read_csv(filepath_distilled, index_col=0).T
    df_small = pd.read_csv(filepath_small_model, index_col=0).T
    df_large = pd.read_csv(filepath_large_model, index_col=0).T
    
    # Get model names from filepaths:
    model_distilled_name = Path(filepath_distilled).stem
    model_small_name = Path(filepath_small_model).stem
    model_large_name = Path(filepath_large_model).stem
    
    # Rename index:
    df_distilled.index = [model_distilled_name]  # type: ignore
    df_small.index = [model_small_name]  # type: ignore
    df_large.index = [model_large_name]  # type: ignore
    
    # Concatenate dataframes and compute RER:
    df = pd.concat([df_distilled, df_small, df_large], axis=0).T
    df["RER (%) - small"] = (df[model_small_name] - df[model_distilled_name]) / df[model_small_name] * 100
    df["RER (%) - large"] = (df[model_large_name] - df[model_distilled_name]) / df[model_large_name] * 100
    
    # Reorder columns:
    df = df[[model_distilled_name, model_small_name, "RER (%) - small", model_large_name, "RER (%) - large"]]
    
    # Post-process:
    if dataset_group is None:
        pass
    elif dataset_group == "esb_librispeech":
        df = post_process_esb_librispeech(df)
    elif dataset_group == "esb_ami":
        df = post_process_esb_ami(df)
    elif dataset_group == "mls":
        df = post_process_mls(df)
    else:
        raise ValueError(f"Invalid dataset group '{dataset_group}'.")
    
    # Convert to LaTeX:
    output = df.round(2).to_latex(column_format="l||c|cc|cc")
    
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
