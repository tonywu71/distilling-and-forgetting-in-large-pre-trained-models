from pathlib import Path

# ------ wandb ------
WANDB_PROJECT = "distilling-and-forgetting-in-large-pre-trained-models"


# ------ Constants ------
DEFAULT_LABEL_COL = "label"
PADDING_IDX = -100
GEN_MAX_LENGTH = 225


# ------ Filepaths ------
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_DIR = Path("outputs")
