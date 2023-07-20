import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env
initialize_env()

from pathlib import Path
from transformers import WhisperTokenizerFast
from datasets import load_from_disk
from dataloader.utils import get_map_funcion_to_restore_missing_special_tokens
from utils.constants import DEFAULT_NUM_PROC


def main(dirpath: str = typer.Argument(..., help="Path to the dataset directory."),
         savepath: str = typer.Argument(..., help="Path to the save directory.")):
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="en", task="transcribe")
    dataset_dict = load_from_disk(dirpath)
    
    dataset_dict = dataset_dict.map(lambda batch: {"teacher_sequences": tokenizer(batch["teacher_text"]).input_ids},
                                    batched=True,
                                    remove_columns=["teacher_text"])
    map_funcion_to_restore_missing_special_tokens = get_map_funcion_to_restore_missing_special_tokens(col="teacher_sequences",
                                                                                                        pretrained_model_name_or_path="openai/whisper-tiny",
                                                                                                        language="en",
                                                                                                        task="transcribe")
    dataset_dict = dataset_dict.map(map_funcion_to_restore_missing_special_tokens, num_proc=DEFAULT_NUM_PROC)
    
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(savepath)
    print("Done!")


if __name__ == "__main__":
    typer.run(main)
