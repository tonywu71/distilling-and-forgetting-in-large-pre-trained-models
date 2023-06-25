import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pathlib import Path
import re
from tqdm.auto import tqdm

import pandas as pd

from utils.constants import DEFAULT_OUTPUT_DIR


def main(dirpath_checkpoints: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False, resolve_path=True)):
    """
    Given a dirpath that contains a bunch of files named `checkpoint-{idx}-fab.csv` (e.g. `checkpoint-500-fab.csv`, `checkpoint-1000-fab.csv`, etc.),
    concatenate all the files into a single CSV file.
    
    The `checkpoint-{idx}-fab.csv` files are the following form:
    ```csv
    Dataset,WER (%)
    librispeech_en_clean,7.69
    librispeech_fr,39.13
    librispeech_pt,37.31
    Average,28.04
    ```
    
    The output CSV file will be of the following form:
    ```csv
    steps,WER librispeech_en_clean (%),WER librispeech_fr (%),WER librispeech_pt (%)
    500,6.37,33.16,
    1000,7.69,39.13,
    ...
    """
    
    # Get all the files in the directory.
    # The files should be named `checkpoint-{idx}-fab.csv`.
    files = [f for f in dirpath_checkpoints.iterdir() if f.is_file() and f.name.endswith(".csv")]
    assert files, f"No CSV files found in `{dirpath_checkpoints}`."
    
    # Initialize the output DataFrame:
    data = {"steps": []}
    
    f = files[0]
    list_datasets = pd.read_csv(f)["Dataset"].tolist()
    
    # Remove `Average`` from list of datasets if exists:
    if "Average" in list_datasets:
        list_datasets.remove("Average")
    
    # Add the datasets to the data dict:
    for dataset in list_datasets:
        data[f"WER {dataset} (%)"] = []
    
    # Create the (empty) DataFrame:
    df_all = pd.DataFrame.from_dict(data)
    
    # Concatenate all the CSV files into a single DataFrame.
    for f in tqdm(files):
        df = pd.read_csv(f)
        
        # Drop the `Average` row:
        df = df[df["Dataset"] != "Average"]
        
        # Get number of step using Regex:
        match = re.search(r"checkpoint-(\d+)-fab.csv", f.name)
        if match:
            steps = int(match.group(1))
        else:
            continue
        
        df_all.loc[len(df_all.index)] = [steps] + df["WER (%)"].tolist()
    
    # Set the `steps` column as int:
    df_all["steps"] = df_all["steps"].astype(int)
    
    # Sort the DataFrame by the `steps` column:
    df_all = df_all.sort_values(by=["steps"])
    
    print(df_all)
    
    # Save the DataFrame to CSV:
    savepath = DEFAULT_OUTPUT_DIR / "report" / "forgetting_wrt_training_steps" / f"{dirpath_checkpoints.name}-eval_forgetting.csv"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(savepath, index=False)
    
    print(f"Saved to `{savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
