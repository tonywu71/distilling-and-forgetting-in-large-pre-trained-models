import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional
from pprint import pprint 

import wandb

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from evaluation.dataset_name_to_dataset_group import DATASET_NAME_TO_DATASET_GROUP
from evaluation.eval_whisper_on_dataset_group import eval_whisper_on_dataset_group
from evaluation.eval_whisper_utils import save_wer_to_csv, log_wer_to_wandb, save_edit_metrics_to_csv, log_edit_metrics_to_wandb
from utils.file_io import extract_exp_name_from_model_path
from utils.constants import DEFAULT_EVAL_BATCH_SIZE, DEFAULT_EVAL_NUM_BEAMS



def main(pretrained_model_name_or_path: str = typer.Argument(..., help="Path to the pretrained model."),
         dataset_name: str = typer.Option(..., help="Name of the dataset to evaluate on."),
         streaming: bool = typer.Option(False, help="Whether to use streaming inference."),
         subset: Optional[List[str]] = typer.Option(None, help="Subset of the ESB dataset to evaluate on."),
         filter_audio_length: bool = typer.Option(False, help="Whether to filter out audio files that are too short or too long. Disabled by default."),
         task: str = typer.Option("transcribe", help="Task to evaluate on."),
         zero_shot: bool = typer.Option(False, help="Whether to use zero-shot inference. Defaults to False."),
         num_beams: int = typer.Option(DEFAULT_EVAL_NUM_BEAMS, help="Number of beams for the ASR pipeline."),
         batch_size: int = typer.Option(DEFAULT_EVAL_BATCH_SIZE, help="Batch size for the ASR pipeline."),
         savepath: Optional[str] = typer.Option(None, help="Filename of the output CSV file. Leave to `None` to use the name of `pretrained_model_name_or_path` as the filename.")) -> None:
    """
    Evaluate the pre-trained Whisper model on a DatasetGroup instance.
    """
    
    assert dataset_name in DATASET_NAME_TO_DATASET_GROUP.keys(), f"Dataset name must be one of {list(DATASET_NAME_TO_DATASET_GROUP.keys())}."
    
    # Load dataset:
    dataset_group: BaseDatasetGroup = DATASET_NAME_TO_DATASET_GROUP[dataset_name](streaming=streaming, subset=subset)
    
    # Create config for wandb:
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "dataset_name": dataset_name,
        "language": dataset_group.language,
        "streaming": streaming,
        "subset": subset,
        "task": task,
        "zero_shot": zero_shot,
        "num_beams": num_beams,
        "batch_size": batch_size
    }
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_EVALUATION"],
               job_type="evaluation",
               tags=[dataset_name],
               name=f"eval_{dataset_name}-{extract_exp_name_from_model_path(pretrained_model_name_or_path)}",
               config=config)
    
    # Print config:
    print("Parameters:")
    pprint(config)
    
    # Load dataset:
    if subset:
        print(f"Subset(s) of {dataset_name}: {subset}")
    
    # Print loaded datasets:
    print(f"Loaded datasets: {list(dataset_group.keys())}")
    
    # If needed, filter out audio files that are too short or too long:
    if filter_audio_length:
        print("Filtering out audio files that are too short or too long...")
        dataset_group.filter_audio_length(verbose=True)
    
    # Evaluate:
    print("Evaluating...")
    df_edit_metrics = eval_whisper_on_dataset_group(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                    ds_group=dataset_group,
                                                    task=task,
                                                    zero_shot=zero_shot,
                                                    batch_size=batch_size,
                                                    num_beams=num_beams)
    
    print("\n-----------------------\n")
    
    print("Results:")
    print(df_edit_metrics)
    
    print("\n-----------------------\n")
    
    
    # Save the WER metrics:
    wer_metrics = df_edit_metrics["WER (%)"]
    
    # Compute the average WER:
    wer_metrics["Average"] = wer_metrics.mean()
    
    # Round the results:
    wer_metrics = wer_metrics.round(2)
    
    
    # Save and log the WER metrics:
    save_wer_to_csv(wer_metrics=wer_metrics,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    dataset_name=dataset_name,
                    savepath=savepath)
    log_wer_to_wandb(wer_metrics)
    
    # Save and log all edit metrics:
    save_edit_metrics_to_csv(df_edit_metrics=df_edit_metrics,
                                pretrained_model_name_or_path=pretrained_model_name_or_path,
                                dataset_name=dataset_name,
                                savepath=savepath)
    log_edit_metrics_to_wandb(df_edit_metrics=df_edit_metrics)
    
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
