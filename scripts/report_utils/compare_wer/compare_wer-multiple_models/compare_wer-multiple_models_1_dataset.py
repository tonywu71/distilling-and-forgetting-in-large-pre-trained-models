import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
from pathlib import Path

import pandas as pd


def main(filepaths: List[str],
         dataset: str = typer.Option(..., help="Dataset to use."),):
    """
    Script that takes multiple CSV outputs from `eval_whisper_on_esb.py` and outputs a comparison table in LaTeX.
    Note that RER is not computed here.
    
    To be used for LaTeX table generation in reports.
    """
    
    list_df = [pd.read_csv(filepath, index_col=0).T for filepath in filepaths]
    list_models = [Path(filepath).stem for filepath in filepaths]
    
    for idx, (df, model) in enumerate(zip(list_df, list_models)):
        list_df[idx].index = [model]  # type: ignore
    
    df = pd.concat(list_df, axis=0)
    
    df = df[dataset]
    
    output = df.round(2).to_latex()
    
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
