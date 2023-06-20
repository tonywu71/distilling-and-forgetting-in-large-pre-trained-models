from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class FinetuneConfig:
    """
    Config class for the Whisper experiments.
    
    Notes:
    - If not defined in the config, `eval_batch_size` will be set to `batch_size`.
    - `is_tokenizer_multilingual` is used to identify the saved/loaded preprocessed datasets
      as there are two different tokenizers (one for English and one for multilingual) and no
      way to know which one was used to preprocess the dataset if a dir checkpoint is provided.
    - `smart_load` is used to load/save the preprocessed dataset from
      `os.environ["PREPROCESSED_DATASETS_DIR"]` to save computation time. Set to False to
      disable this feature.
    """
    experiment_name: str
    lang_name: str
    task: str
    pretrained_model_name_or_path: str
    is_tokenizer_multilingual: bool
    model_dir: str
    freeze_encoder: bool
    freeze_decoder: bool
    batch_size: int
    gradient_accumulation_steps: int  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-accumulation
    gradient_checkpointing: bool  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing
    dataset_name: str
    optim: str
    learning_rate: float
    warmup_steps: int
    eval_steps: int
    generation_num_beams: int
    save_steps: int
    logging_steps: int
    num_train_epochs: int
    
    # ======== Optional (data preprocessing) ========
    data_augmentation: bool = False
    lowercase: bool = True  # set to False if and only if the text is not fully uppercased
    
    # ======== Optional (training) ========
    zero_shot: bool = True
    eval_batch_size: Optional[int] = None
    eval_accumulation_steps: Optional[int] = None  # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps
    save_total_limit: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    
    # ======== Other ========
    smart_load: bool = True
    force_reprocess_dataset: bool = False
    eval_first_step: bool = True
    log_preds_to_wandb: bool = True
    log_raw_str: bool = False
    n_samples_per_wandb_logging_step: int = 8
    
    experimental_train_implicit_lm: bool = False
    
    
    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        
        # Set defaults:
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if self.early_stopping_patience is None:
            self.early_stopping_patience = -1
        
        # Sanity checks:
        assert self.save_total_limit is None or self.save_total_limit >= 2, \
            "The `save_total_limit` must be at least 2, or None."
    
    
    
    @staticmethod
    def from_yaml(config_file: str) -> "FinetuneConfig":
        """Parse the YAML config file and return a Config object"""
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert types:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])
        
        # Fix paths:
        if config_dict["model_dir"].endswith("/"):
            # The model_dir must end with a slash:
            config_dict["model_dir"] = config_dict["model_dir"] + "/"
        
        return FinetuneConfig(**config_dict)
