import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from toolz import dicttoolz
from safetensors import safe_open
import pandas as pd

from utils.constants import DEFAULT_OUTPUT_DIR


def main(dirpath_ewc: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False, help="Path to the EWC directory.")):
    filepath_ewc_fisher = dirpath_ewc / "ewc_fisher_params.safetensors"
    assert filepath_ewc_fisher.exists(), f"`filepath_ewc_fisher` does not exist: {filepath_ewc_fisher}"
    
    ewc_fisher_params = {}
    with safe_open(filepath_ewc_fisher, framework="pt") as f:
        for k in f.keys():
            ewc_fisher_params[k] = f.get_tensor(k)
    
    ewc_fisher_params = pd.Series(dicttoolz.valmap(lambda x: x.mean().item(), ewc_fisher_params))
    savepath = DEFAULT_OUTPUT_DIR / "ewc_fisher_params.csv"
    ewc_fisher_params.to_csv(savepath)
    print(f"Saved the EWC Fisher parameters to {savepath}.")
    
    return

if __name__ == "__main__":
    typer.run(main)
