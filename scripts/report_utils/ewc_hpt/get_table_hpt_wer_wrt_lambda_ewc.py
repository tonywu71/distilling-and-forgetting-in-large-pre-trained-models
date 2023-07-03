import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pathlib import Path
import pandas as pd


def main(dirpath: Path = typer.Argument(..., file_okay=False, dir_okay=True)):
    assert dirpath.is_dir(), f"{dirpath} is not a directory"
    
    # Print the number of files found:
    list_filepaths = sorted(list(dirpath.glob("*")))
    print(f"Found {len(list_filepaths)} files.")
    list_df = [pd.read_csv(filepath).assign(lambda_ewc=filepath.stem) for filepath in dirpath.glob("*")]
    
    # Load and concatenate all CSV files:
    list_df = [pd.read_csv(filepath).assign(lambda_ewc=filepath.stem) for filepath in dirpath.glob("*")]
    df = pd.concat(list_df).reset_index(drop=True)
    df["lambda_ewc"] = df["lambda_ewc"].str.extract(r'lambda_(.+)').astype(float)
    df = df[df["lambda_ewc"]!= 0]  # drop rows where lambda = 0
    
    # Print the DataFrame:
    print("Result:")
    print(df.set_index(["Dataset", "lambda_ewc"]).sort_index(level=0))
    print()
    
    # Generate the LaTeX table:
    ser = df.set_index(["Dataset", "lambda_ewc"]).sort_index(level=0)
    output = ser.to_latex(float_format="%.2f", escape=True)
        
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
