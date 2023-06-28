import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env
initialize_env()

from pathlib import Path
from typing import Dict
import torch
from datasets import load_from_disk


def main(dirpath: str = typer.Argument(..., help="Path to the dataset directory."),
         savepath: str = typer.Argument(..., help="Path to the save directory.")):
    dataset_dict = load_from_disk(dirpath)

    def map_fct(labels: torch.LongTensor) -> Dict[str, torch.LongTensor]:
        labels = torch.cat([labels[0:1], torch.LongTensor([50259, 50359]), labels[1:]])  # add missing tokens for <EN> and <TRANSCRIBE>
        return {"labels": labels}
    
    for split in dataset_dict:
        print(f"Preprocessing the {split} set...")
        dataset_dict[split] = dataset_dict[split].map(map_fct, input_columns=["labels"], num_proc=4)
    
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(savepath)
    print("Done!")


if __name__ == "__main__":
    typer.run(main)
