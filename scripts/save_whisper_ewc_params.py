import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from pathlib import Path
from safetensors.torch import save_file

import wandb

from trainer.ewc_estimation import get_ewc_params_for_whisper
from utils.file_io import extract_exp_name_from_model_path
from utils.constants import CHECKPOINTS_DIR


def get_dirpath_ewc_params(pretrained_model_name_or_path: str,
                           language: str,
                           task: str,
                           dataset_name: str,
                           split: str) -> str:
    """
    Get the directory path to save the EWC params.
    Handles the edge case where the pretrained model is a path to a HuggingFace Hub model.
    """
    if pretrained_model_name_or_path.startswith("openai/whisper-"):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)        
        prefix = CHECKPOINTS_DIR / pretrained_model_name_or_path.split("/")[-1]
    else:
        prefix = Path(pretrained_model_name_or_path)
        assert prefix.is_dir(), f"Invalid `pretrained_model_name_or_path`: {pretrained_model_name_or_path}"
    
    dirpath = os.path.join(prefix, language, task, dataset_name, split)
    return dirpath


def main(pretrained_model_name_or_path: str = typer.Argument(..., help="The name or path of the pretrained model."),
         language: str = typer.Option(..., help="The language of the pretrained model."),
         task: str = typer.Option("transcribe", help="The task of the pretrained model."),
         dataset_name: str = typer.Option(..., help="The name of the dataset."),
         split: str = typer.Option("train", help="The split of the dataset."),
         skip_lowercase: bool = typer.Option(False, help="Whether to skip the lowercase preparation of the dataset."),
         batch_size: int = typer.Option(32, help="The batch size for the dataloader.")):
    
    # Create config for wandb:
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "language": language,
        "task": task,
        "dataset_name": dataset_name,
        "split": split,
        "skip_lowercase": skip_lowercase,
        "batch_size": batch_size
    }
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_OTHERS"],
               job_type="save-ewc-params",
               tags=[dataset_name],
               name=f"save_ewc_params-{extract_exp_name_from_model_path(pretrained_model_name_or_path)}-{task}-{language}",
               config=config)
    
    # Get the EWC params:
    mean_params, fisher_params = get_ewc_params_for_whisper(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                            language=language,
                                                            task=task,
                                                            dataset_name=dataset_name,
                                                            split=split,
                                                            batch_size=batch_size,
                                                            lowercase=not(skip_lowercase))
    
    # Save the EWC params:
    dirpath = get_dirpath_ewc_params(pretrained_model_name_or_path,
                                     language=language,
                                     task=task,
                                     dataset_name=dataset_name,
                                     split=split)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    
    mean_params_savepath = os.path.join(dirpath, "ewc_mean_params.safetensors")
    save_file(mean_params, mean_params_savepath)
    print(f"Saved the EWC mean parameters to `{mean_params_savepath}`.")
    
    fisher_params_savepath = os.path.join(dirpath, "ewc_fisher_params.safetensors")
    save_file(fisher_params, fisher_params_savepath)
    print(f"Saved the EWC Fisher parameters to `{fisher_params_savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
