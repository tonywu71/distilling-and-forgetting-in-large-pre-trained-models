import typer

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from pprint import pprint

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional

import wandb

from dataloader.datasets.ami_test import AMITestSet
from evaluation.eval_whisper_on_dataset import eval_whisper_on_dataset
from utils.file_io import extract_exp_name_from_model_path, extract_output_savepath_from_model_path



def main(pretrained_model_name_or_path: str=typer.Argument(..., help="Path to the pretrained model or its name in the HuggingFace Hub."),
         streaming: bool=typer.Option(False, help="Whether to use streaming inference."),
         batch_size: int=typer.Option(16, help="Batch size for the ASR pipeline."),
         savepath: Optional[str]=typer.Option(
             None, help="Filename of the output CSV file. Leave to `None` to use the name of `pretrained_model_name_or_path` as the filename.")) -> None:
    """
    Evaluate the whisper model on the LibriSpeech clean test set.
    Note that only greedy decoding is supported for now.
    """
    
    # Set up the parameters:
    task = "transcribe"
    subset = None  # there is only one dataset in this dataset group so we don't need to specify a subset
    
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "task": task,
        "dataset": "librispeech",
        "streaming": streaming,
        "subset": subset,
        "batch_size": batch_size,
    }
    
    print("Parameters:")
    pprint(config)
    
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT"],
               job_type="evaluation",
               tags=["ami_test"],
               name=f"eval_librispeech_clean_test-{extract_exp_name_from_model_path(pretrained_model_name_or_path)}",
               config=config)
    
    
    # Load dataset:
    if subset:
        print(f"Subset(s) of LibriSpeech: {subset}")
        
    ami_dataset = AMITestSet(streaming=streaming, subset=subset)
    print(f"Loaded datasets: {list(ami_dataset.keys())}")
    
    
    # Preprocess:
    print("Preprocessing datasets...")
    ami_dataset.preprocess_datasets(normalize=True)
    
    
    # Evaluate:
    print("Evaluating...")
    results = eval_whisper_on_dataset(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                      ds_group=ami_dataset,
                                      batch_size=batch_size,
                                      task=task)
    
    print("Results:")
    print(results)
    
    
    print()
    
    
    # Save results:
    if savepath is None:
        savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path)  + "-ami.csv"
    
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(f"{savepath}")
    print(f"Results saved to `{savepath}`.")
    
    
    # Log results to W&B:
    barplot = wandb.plot.bar(wandb.Table(dataframe=results.to_frame().reset_index()),  # type: ignore
                             label=results.index.name,
                             value=str(results.name),
                             title="Per dataset WER (%)")
    wandb.log({"wer_for_dataset_group": barplot})
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
