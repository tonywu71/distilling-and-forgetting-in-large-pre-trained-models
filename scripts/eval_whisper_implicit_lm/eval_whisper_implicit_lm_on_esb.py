from pathlib import Path
from pprint import pprint
import typer

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional

import wandb

from dataloader.datasets.esb_dataset import ESBDataset
from evaluation.eval_whisper_implicit_lm_on_dataset import eval_whisper_implicit_lm_on_dataset
from utils.file_io import extract_experiment_name, extract_savepath


def main(pretrained_model_name_or_path: str,
         streaming: bool=typer.Option(False, help="Whether to use streaming inference."),
         load_full: bool=typer.Option(False, help="Whether to load the full ESB dataset (non diagnostic)."),
         subset: Optional[List[str]]=typer.Option(None, help="Subset of the ESB dataset to evaluate on."),
         savepath: Optional[str]=typer.Option(
             None, help="Filename of the output CSV file. Leave to `None` to use the name of `pretrained_model_name_or_path` as the filename.")) -> None:
    """
    Evaluate the whisper implicit language model on the ESB benchmark (diagnostic by default).
    Note that only greedy decoding is supported for now.
    """
    
    load_diagnostic = not load_full
    
    # Set up the parameters:
    language = "english"
    task = "transcribe"
    
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "language": language,
        "task": task,
        "dataset": "esb",
        "streaming": streaming,
        "load_diagnostic": load_diagnostic,
        "subset": subset
    }
    
    print("Parameters:")
    pprint(config)
    
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT"],
               job_type="evaluation",
               name=f"eval_esb-{extract_experiment_name(pretrained_model_name_or_path)}-implicit_lm",
               config=config)
    
    
    # Load dataset:
    if subset:
        print(f"Subset(s) of ESB: {subset}")
        
    esb_dataset = ESBDataset(streaming=streaming,
                             load_diagnostic=load_diagnostic,
                             subset=subset)
    print(f"Loaded datasets: {list(esb_dataset.keys())}")
    
    
    # Preprocess:
    print("Preprocessing datasets...")
    esb_dataset.preprocess_datasets(normalize=True)
    
    
    # Evaluate:
    print("Evaluating...")
    results = eval_whisper_implicit_lm_on_dataset(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                  ds_group=esb_dataset,
                                                  task=task)
    
    print("Results:")
    print(results)
    
    
    print()
    
    
    # Save results:
    if savepath is None:
        savepath = extract_savepath(pretrained_model_name_or_path) + "-implicit_lm" + "-esb.csv"
    
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(f"{savepath}")
    print(f"Results saved to `{savepath}`.")
    
    
    # Log results to W&B:
    barplot = wandb.plot.bar(wandb.Table(dataframe=results.to_frame().reset_index()),  # type: ignore
                             label=results.index.name,
                             value=str(results.name),
                             title="Per dataset perplexity (%)")
    wandb.log({"perplexity_for_dataset_group": barplot})
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
