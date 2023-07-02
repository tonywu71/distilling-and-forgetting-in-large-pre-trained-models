import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
from pathlib import Path
from datetime import datetime

import pandas as pd

from utils.constants import DEFAULT_OUTPUT_DIR


def main(filepaths: List[str]):
    """
    Script that takes multiple CSV outputs from `eval_whisper_on_esb.py` and outputs a concatenated CSV table.
    Note that RER is not computed here.
    """
    
    list_df = [pd.read_csv(filepath, index_col=0).T for filepath in filepaths]
    list_models = [Path(filepath).stem for filepath in filepaths]
    
    for idx, (df, model) in enumerate(zip(list_df, list_models)):
        list_df[idx].index = [model]  # type: ignore
    
    df = pd.concat(list_df, axis=0).T
    
    # Save concatenated CSV table:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    savepath = (DEFAULT_OUTPUT_DIR / f"compare_multiple_models_-{timestamp}").with_suffix(".csv")
    savepath.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(savepath)
    print(f"Saved concatenated CSV table to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
