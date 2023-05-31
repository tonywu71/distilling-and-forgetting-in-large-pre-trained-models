from pathlib import Path

# ------ Config ------
DEFAULT_ENV_CONFIG_FILEPATH = "configs/env_config.yaml"
CHECKPOINTS_DIRNAME = "checkpoints"
K_BEAM_SEARCH_DIRNAME = "k_beam_search"
DEFAULT_NUM_PROC = 8  # see https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#hardware


# ------ Constants ------
DEFAULT_LABEL_STR_COL = "text"
DEFAULT_LABEL_TOKENIZED_COL = "labels"  # default column name used by the HuggingFace Trainer
PADDING_IDX = -100
GEN_MAX_LENGTH = 225


# ------ Filepaths ------
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_DIR = Path("outputs")
