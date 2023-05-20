import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

from utils.constants import DEFAULT_OUTPUT_DIR


def main(wer_filepath: str, ppl_filepath: str, model_name: str):
    df_wer = pd.read_csv(wer_filepath)
    df_ppl = pd.read_csv(ppl_filepath)
    df = df_wer.merge(df_ppl, on="Dataset")
    df["Model"] = model_name
    
    # Drop rows where "Dataset" contains "Average":
    df = df[~df["Dataset"].str.contains("Average")]
    
    # Save concatenated CSV table:
    savepath = (DEFAULT_OUTPUT_DIR / "report" / "plot_wer_wrt_perplexity" / f"wer_and_ppl-{model_name}").with_suffix(".csv")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(savepath)
    print(f"Saved concatenated CSV table to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
