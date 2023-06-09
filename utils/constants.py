from pathlib import Path

# ------ Config ------
DEFAULT_ENV_CONFIG_FILEPATH = "configs/env_config.yaml"
CHECKPOINTS_DIRNAME = "checkpoints"
DEFAULT_NUM_PROC = 8  # see https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#hardware


# ------ Constants ------
DEFAULT_LABEL_STR_COL = "text"
DEFAULT_LABEL_TOKENIZED_COL = "labels"  # default column name used by the HuggingFace Trainer
DEFAULT_EVAL_NUM_BEAMS = 5
LOSS_MASK_IDX = -100  # see https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.forward for the `labels` argument
GEN_MAX_LENGTH = 225

# ------ Filepaths ------
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_DIR = Path("outputs")
