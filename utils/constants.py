from pathlib import Path

# ------ Config ------
DEFAULT_ENV_CONFIG_FILEPATH = "configs/env_config.yaml"


# ------ wandb ------
DEFAULT_N_SAMPLES_PER_WANDB_LOGGING_STEP = 8


# ------ Constants ------
DEFAULT_LABEL_STR_COL = "text"
DEFAULT_LABEL_TOKENIZED_COL = "labels"  # default column name used by the HuggingFace Trainer
PADDING_IDX = -100
GEN_MAX_LENGTH = 225


# ------ Filepaths ------
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_DIR = Path("outputs")
