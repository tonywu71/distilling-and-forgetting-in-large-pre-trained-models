from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import yaml


AVAILABLE_KD_METHODS = ["word_level", "seq_level_mode", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]


@dataclass
class DistilConfig:
    """
    Config class for distillation experiments.
    
    Notes:
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
    method: Literal["word_level", "seq_level_mode", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]
    teacher_model_name_or_path: str
    student_model_name_or_path: str
    is_tokenizer_multilingual: bool
    model_dir: str
    freeze_encoder: bool
    freeze_decoder: bool
    batch_size: int
    gradient_accumulation_steps: int  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-accumulation
    eval_accumulation_steps: Optional[int]  # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps
    gradient_checkpointing: bool  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing
    data_augmentation: bool
    dataset_name: str
    force_reprocess_dataset: bool
    optim: str
    learning_rate: float
    warmup_steps: int
    eval_steps: int
    generation_num_beams: int
    save_steps: int
    save_total_limit: Optional[int]
    logging_steps: int
    num_train_epochs: int
    early_stopping_patience: Optional[int]
    
    
    # ======== Knowledge distillation hyperparameters ========
    # TODO: Set defaults for these hyperparameters after running some experiments.
    # General:
    ce_alpha: float = 0.5
    
    # `word_level`:
    temperature: Optional[float] = None
    
    # Sequence-level (`seq_level_mode`, `seq_level_k_best_uniform`, `seq_level_k_best_ranked`)
    distillation_num_beams: Optional[int] = None
    
    # `seq_level_k_best_ranked`:
    decay_beta: Optional[float] = None
    
    
    # ======== Other ========
    smart_load: bool = True
    log_preds_to_wandb: bool = True
    n_samples_per_wandb_logging_step: int = 8
    log_raw_str: bool = False
    
    
    
    def __post_init__(self) -> None:
        """Post-initialization checks."""
        
        assert self.save_total_limit is None or self.save_total_limit >= 2, \
            "The `save_total_limit` must be at least 2, or None."
        
        self._validate_distillation_args()
    
    
    def _validate_distillation_args(self) -> None:
        """Validate the distillation arguments."""
        
        assert self.method in AVAILABLE_KD_METHODS, \
            f"Invalid distillation method `{self.method}`. Available methods: {AVAILABLE_KD_METHODS}."
        
        if self.method == "word_level":
            assert self.temperature is not None, \
                "The `temperature` must be set for `word_level` distillation."
        if self.method in ["seq_level_mode", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]:
            assert self.distillation_num_beams is not None, \
                "The `distillation_num_beams` must be set for sequence-level distillation."
        if self.method == "seq_level_k_best_ranked":
            assert self.decay_beta is not None, \
                "The `decay_beta` must be set for `seq_level_k_best_ranked` distillation."
    
    
    @staticmethod
    def from_yaml(config_file: str) -> "DistilConfig":
        """Parse the YAML config file and return a DistillConfig object"""
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert types:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])
        
        # Fix paths:
        if config_dict["model_dir"] and not config_dict["model_dir"].endswith("/"):
            # The model_dir must end with a slash:
            config_dict["model_dir"] = config_dict["model_dir"] + "/"
        
        # Set defaults:
        if config_dict["early_stopping_patience"] is None:
            config_dict["early_stopping_patience"] = -1
        
        return DistilConfig(**config_dict)
