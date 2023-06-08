import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Literal
from pathlib import Path
import yaml
from dataclasses import asdict
import pandas as pd

from utils.finetune_config import FinetuneConfig
from utils.distil_config import DistilConfig


def main(config_file: str = typer.Argument(..., help="Path to model config file (.yaml)"),
         kind: str = typer.Option("other", help="Kind of config file to convert to LaTeX table. " + \
                                                                    "Set to `other` if not a `FinetuneConfig` or `DistilConfig` file.")):
    """
    Script to convert a YAML config file to a LaTeX table.
    To be used for LaTeX table generation in reports.
    """
    if kind == "finetune":
        config = FinetuneConfig.from_yaml(config_file)
        ser = pd.Series(asdict(config), name="Value")
    elif kind == "distil":
        config = DistilConfig.from_yaml(config_file)
        ser = pd.Series(asdict(config), name="Value")
    elif kind == "other":
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            ser = pd.Series(config, name="Value")
    else:
        raise ValueError(f"Invalid `kind` argument: {kind}. Must be one of `finetune`, `distil`, or `other`.")
    
    ser.index.name = "Hyperparameter"
    
    output = ser.to_latex(column_format="l|c")
    
    print("```latex")
    print(output + "```")
    return


if __name__ == "__main__":
    typer.run(main)
