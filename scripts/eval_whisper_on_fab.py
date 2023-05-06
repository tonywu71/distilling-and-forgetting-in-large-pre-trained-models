from pathlib import Path
from pprint import pprint
import typer

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional

import wandb

from dataloader.datasets.fab_dataset import FABDataset
from evaluation.eval_whisper_on_dataset import eval_whisper_on_dataset
from utils.file_io import extract_savepath_from_model_filepath



def main(pretrained_model_name_or_path: str=typer.Argument(..., help="Path to the pretrained model or its name in the HuggingFace Hub."),
         language: str=typer.Option("english", help="Language of the model."),
         streaming: bool=typer.Option(False, help="Whether to use streaming inference."),
         subset: Optional[List[str]]=typer.Option(None, help="Subset of the FAB dataset to evaluate on."),
         batch_size: int=typer.Option(16, help="Batch size for the ASR pipeline."),
         savepath: Optional[str]=typer.Option(
             None, help="Filename of the output CSV file. Leave to `None` to use the name of `pretrained_model_name_or_path` as the filename.")) -> None:
    """
    Evaluate the whisper model on the ESB benchmark.
    Note that only greedy decoding is supported for now.
    """
    
    # Set up the parameters:
    task = "transcribe"
    
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "language": language,
        "task": task,
        "dataset": "fab",
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
               name=f"eval_fab-{extract_savepath_from_model_filepath(pretrained_model_name_or_path).stem}",
               config=config)
    
    
    # Load dataset:
    if subset:
        print(f"Subset(s) of ESB: {subset}")
        
    fab_dataset = FABDataset(streaming=streaming, subset=subset)
    print(f"Loaded datasets: {list(fab_dataset.keys())}")
    
    
    # Evaluate:
    print("Evaluating...")
    # TODO: Figure out how to handle the multilingual case.
    results = eval_whisper_on_dataset(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                      ds_group=fab_dataset,
                                      batch_size=batch_size,
                                      language=language,
                                      task=task)
    
    print("Results:")
    print(results)
    
    
    print()
    
    
    # Save results:
    if savepath is None:
        savepath = str(extract_savepath_from_model_filepath(pretrained_model_name_or_path))
    
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(f"{savepath}")
    print(f"Results saved to `{savepath}`.")
    
    
    # Log results to W&B:
    barplot = wandb.plot.bar(wandb.Table(dataframe=results.to_frame().reset_index()),  # type: ignore
                             label=results.index.name,
                             value=str(results.name),
                             title="Per dataset WER (%)")
    wandb.log({"wer_for_esb_dataset": barplot})
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
