import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from pathlib import Path
import pickle

from trainer.ewc_estimation import get_ewc_params_for_whisper
from utils.constants import EWC_PARAMS_VANILLA


def get_dirpath_ewc_params(pretrained_model_name_or_path: str) -> str:
    """
    Get the directory path to save the EWC params.
    Handles the edge case where the pretrained model is a path to a HuggingFace Hub model.
    """
    if pretrained_model_name_or_path.startswith("openai/whisper-"):
        EWC_PARAMS_VANILLA.mkdir(parents=True, exist_ok=True)        
        return str(EWC_PARAMS_VANILLA / pretrained_model_name_or_path.split("openai/whisper-")[-1])
    else:
        return pretrained_model_name_or_path


def main(pretrained_model_name_or_path: str = typer.Argument(..., help="The name or path of the pretrained model."),
         language: str = typer.Option(..., help="The language of the pretrained model."),
         task: str = typer.Option("transcribe", help="The task of the pretrained model."),
         dataset_name: str = typer.Option(..., help="The name of the dataset."),
         batch_size: int = typer.Option(32, help="The batch size for the dataloader.")):
    
    # Get the EWC params:
    mean_params, fisher_params = get_ewc_params_for_whisper(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                            language=language,
                                                            task=task,
                                                            dataset_name=dataset_name,
                                                            batch_size=batch_size)
    
    # Dump the EWC params as pickle files:
    dirpath = get_dirpath_ewc_params(pretrained_model_name_or_path)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(dirpath, "mean_params.pkl"), "wb") as f:
        pickle.dump(mean_params, f)
        print(f"Dumped EWC mean params to `{os.path.join(dirpath, 'mean_params.pkl')}`.")
    
    with open(os.path.join(dirpath, "fisher_params.pkl"), "wb") as f:
        pickle.dump(fisher_params, f)
        print(f"Dumped EWC fisher params to `{os.path.join(dirpath, 'fisher_params.pkl')}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
