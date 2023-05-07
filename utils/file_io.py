from pathlib import Path
from utils.constants import DEFAULT_OUTPUT_DIR, CHECKPOINTS_DIRNAME


def extract_experiment_name(model_filepath: str) -> str:
    """
    Extract the conventional experiment name from a model path.
    
    Examples:
    - "tw581/checkpoints/whisper_small/librispeech_clean_100h/checkpoint-200" -> "whisper_small/librispeech_clean_100h/checkpoint-200"
    - "openai/whisper-tiny.en" -> "openai/whisper-tiny-en"
    """
    path = Path(model_filepath)
    
    if CHECKPOINTS_DIRNAME in path.parts:  # if the path is a checkpoint...
        experiment_name = path.relative_to(CHECKPOINTS_DIRNAME).as_posix()
    else:  # if the path is a model name from the HuggingFace Hub...
        experiment_name = path.name.replace(".", "-")
    
    return experiment_name


def extract_savepath(model_filepath: str) -> str:
    """
    Extract the model savepath from a model path.
    
    Examples:
    - "tw581/checkpoints/whisper_small/librispeech_clean_100h/checkpoint-200" -> "outputs/whisper_small/librispeech_clean_100h/checkpoint-200"
    - "openai/whisper-tiny.en" -> "outputs/whisper-tiny-en"
    """
    path = Path(model_filepath)
    
    if CHECKPOINTS_DIRNAME in path.parts:  # if the path is a checkpoint...
        savepath = (DEFAULT_OUTPUT_DIR / path.relative_to(CHECKPOINTS_DIRNAME)).as_posix()
    else:  # if the path is a model name from the HuggingFace Hub...
        savepath = (DEFAULT_OUTPUT_DIR / path.name.replace(".", "-")).as_posix()
    
    return savepath
