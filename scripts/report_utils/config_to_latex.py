import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import asdict
import pandas as pd

from utils.config import load_yaml_config


def main(config_file: str=typer.Argument(..., help="Path to model config file (.yaml)")):
    """
    Script to convert a YAML config file to a LaTeX table.
    To be used for LaTeX table generation in reports.
    """
    config = load_yaml_config(config_file)

    ser = pd.Series(asdict(config), name="Value")
    ser.index.name = "Hyperparameter"
    
    output = ser.to_latex(column_format="l|c")
    
    print("```latex")
    print(output + "```")
    return


if __name__ == "__main__":
    typer.run(main)
