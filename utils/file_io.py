from pathlib import Path
from utils.constants import DEFAULT_OUTPUT_DIR, CHECKPOINTS_DIRNAME


def extract_savepath_from_model_filepath(model_filepath: str) -> Path:
    """
    Extract the model savepath from a model path.
    """
    path = Path(model_filepath)
    
    if CHECKPOINTS_DIRNAME in path.parts:  # if the path is a checkpoint...
        # Example: test/checkpoints/whisper_small/librispeech_100h/checkpoint-200 -> whisper_small/librispeech_100h/checkpoint-200.csv
        savepath = DEFAULT_OUTPUT_DIR / path.relative_to(CHECKPOINTS_DIRNAME).with_suffix(".csv")
    else:  # if the path is a model name from the HuggingFace Hub...
        savepath = (DEFAULT_OUTPUT_DIR / path.name.replace(".", "-")).with_suffix(".csv")  # e.g. "openai/whisper-tiny.en" -> "whisper-tiny-en.csv"
    
    return savepath
