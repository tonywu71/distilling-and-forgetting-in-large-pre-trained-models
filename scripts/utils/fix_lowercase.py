import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env
initialize_env()
ß
from pathlib import Path
from typing import Dict, Any
from transformers import WhisperTokenizerFast
from datasets import load_from_disk


def main(dirpath: str = typer.Argument(..., help="Path to the dataset directory."),
         savepath: str = typer.Argument(..., help="Path to the save directory.")):
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="en", task="transcribe")
    dataset_dict = load_from_disk(dirpath)

    def map_fct(batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["text"] = batch["text"].lower()
        # Encode from target text to label ids:
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch
    
    for split in dataset_dict:
        print(f"Preprocessing the {split} set...")
        dataset_dict[split] = dataset_dict[split].map(map_fct, batched=True, batch_size=256)
    
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(savepath)
    print("Done!")


if __name__ == "__main__":
    typer.run(main)
