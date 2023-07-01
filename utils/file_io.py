import os
from pathlib import Path
from datetime import datetime

from utils.finetune_config import FinetuneConfig
from utils.distil_config import DistilConfig
from utils.constants import DEFAULT_OUTPUT_DIR, CHECKPOINTS_DIR


def fix_model_dir_conflicts(config: FinetuneConfig | DistilConfig) -> None:
    """
    If `config.model_dir` is an existing directory, a timestamp will be added to the model directory
    to avoid overwriting previous models.
    
    This method changes the value of `config.model_dir` in-place.
    """
    
    if Path(config.model_dir).is_dir() and os.listdir(config.model_dir):
        # Get the current date and time
        print (f"Model directory `{config.model_dir}` is not empty. A timestamp will be added to the model directory.")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        old_model_dir = Path(config.model_dir)
        new_model_dir = old_model_dir.with_name(f"{old_model_dir.name}-{timestamp}")
        config.model_dir = new_model_dir.as_posix()
        print (f"New model directory: `{config.model_dir}`.")
    
    return


def extract_exp_name_from_model_path(model_filepath: str) -> str:
    """
    Extract the conventional experiment name from a model path.
    
    Examples:
    - "tw581/checkpoints/whisper_small/librispeech_clean_100h/checkpoint-200" -> "whisper_small/librispeech_clean_100h/checkpoint-200"
    - "openai/whisper-tiny.en" -> "openai/whisper-tiny-en"
    """
    path = Path(model_filepath)
    
    if CHECKPOINTS_DIR in path.parts:  # if the path is a checkpoint...
        experiment_name = path.relative_to(CHECKPOINTS_DIR).as_posix()
    else:  # if the path is a model name from the HuggingFace Hub...
        experiment_name = path.name.replace(".", "-")
    
    return experiment_name


def extract_output_savepath_from_model_path(model_filepath: str) -> str:
    """
    Extract the model savepath from a model path.
    
    Examples:
    - "tw581/checkpoints/whisper_small/librispeech_clean_100h/checkpoint-200" -> "outputs/whisper_small/librispeech_clean_100h/checkpoint-200"
    - "openai/whisper-tiny.en" -> "outputs/whisper-tiny-en"
    """
    path = Path(model_filepath)
    
    if CHECKPOINTS_DIR in path.parts:  # if the path is a checkpoint...
        savepath = (DEFAULT_OUTPUT_DIR / path.relative_to(CHECKPOINTS_DIR)).as_posix()
    else:  # if the path is a model name from the HuggingFace Hub...
        savepath = (DEFAULT_OUTPUT_DIR / path.name.replace(".", "-")).as_posix()
    
    return savepath
