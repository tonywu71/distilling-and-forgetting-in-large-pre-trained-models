from pathlib import Path

# ------ Env config ------
DEFAULT_ENV_CONFIG_FILEPATH = "configs/env_config.yaml"
LIST_ENV_VARS = [
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    "HF_MODULES_CACHE",
    "CACHE_DIR_LIBRISPEECH",
    "CACHE_DIR_AMI",
    "CACHE_DIR_ESB",
    "CACHE_DIR_ESB_DIAGNOSTIC",
    "CACHE_DIR_MLS",
    "WANDB_PROJECT_TRAINING",
    "WANDB_PROJECT_EVALUATION",
    "WANDB_PROJECT_OTHERS",
    "WANDB_CACHE_DIR",
    "PREPROCESSED_DATASETS_DIR",
    "K_BEAM_SEARCH_CACHE_DIR"
]


# ------ Config ------
CHECKPOINTS_DIR = Path("checkpoints/")
DEFAULT_NUM_PROC = 8  # see https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#hardware


# ------ Constants ------
DEFAULT_LABEL_STR_COL = "text"
DEFAULT_LABEL_TOKENIZED_COL = "labels"  # default column name used by the HuggingFace Trainer

MIN_INPUT_LENGTH = 0.0
MAX_INPUT_LENGTH = 30.0

DEFAULT_EVAL_BATCH_SIZE = 256  # works for Whisper tiny
DEFAULT_EVAL_NUM_BEAMS = 1

LOSS_MASK_IDX = -100  # see https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.forward for the `labels` argument
GEN_MAX_LENGTH = 225

# ------ Filepaths ------
DEFAULT_LOG_DIR = Path("logs/")
DEFAULT_OUTPUT_DIR = Path("outputs/")
