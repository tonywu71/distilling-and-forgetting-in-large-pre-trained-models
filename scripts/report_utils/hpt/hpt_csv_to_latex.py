import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd


def main(hpt_filepath: str):
    df = pd.read_csv(hpt_filepath, dtype={"learning_rate": str}, index_col="idx")
    
    # Drop all incomplete rows:
    df = df.dropna(axis=0, how="any")
    
    # Convert to LaTeX:
    output = df.round(2).to_latex()
    
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
