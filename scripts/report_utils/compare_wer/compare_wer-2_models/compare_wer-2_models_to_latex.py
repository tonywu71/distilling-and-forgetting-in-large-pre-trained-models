from ast import Not
import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional
from pathlib import Path

import pandas as pd


def post_process_esb_librispeech(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["librispeech_clean", "librispeech_other"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["In-domain average", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Out-of-domain average", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df


def post_process_esb_ami(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["ami"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["In-domain average", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Out-of-domain average", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df


def post_process_mls(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["english"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["English", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Other languages", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df
    


def main(filepath_1: str=typer.Argument(..., help="Path to first CSV file."),
         filepath_2: str=typer.Argument(..., help="Path to second CSV file."),
         dataset_group: Optional[str]=typer.Option(None, help="Dataset group to use. Either 'esb_librispeech', 'esb_ami', or 'mls'.")):
    """
    Script that takes 2 CSV outputs from `eval_whisper_on_esb.py` and outputs a comparison table in LaTeX.
    Use the `dataset_group` option to specify the dataset group to use for additional statistics.ß
    
    To be used for LaTeX table generation in reports.
    """
    
    df_1 = pd.read_csv(filepath_1, index_col=0).T
    df_2 = pd.read_csv(filepath_2, index_col=0).T
    
    model_1 = Path(filepath_1).stem
    model_2 = Path(filepath_2).stem
    
    df_1.index = [model_1]  # type: ignore
    df_2.index = [model_2]  # type: ignore
    
    df = pd.concat([df_1, df_2], axis=0).T
    df["RER (%)"] = (df[model_1] - df[model_2]) / df[model_1] * 100
    
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
    
    output = df.round(2).to_latex(column_format="l|cc|c")
    
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
