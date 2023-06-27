from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import yaml


AVAILABLE_KD_METHODS = ["word_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]


@dataclass
class DistilConfig:
    """
    Config class for distillation experiments.
    
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
    method_distil: Literal["word_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]
    teacher_model_name_or_path: str
    student_model_name_or_path: str
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
    zero_shot_eval: bool = False
    eval_batch_size: Optional[int] = None
    eval_accumulation_steps: Optional[int] = None  # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps
    save_total_limit: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    save_final_model: bool = True
    
    
    # ======== Knowledge distillation hyperparameters ========
    # General:
    alpha_ce: float = 0.5
    
    # `word_level`:
    temperature: float = 2
    
    # Sequence-level (`seq_level_k_best_uniform`, `seq_level_k_best_ranked`)
    distillation_num_beams: Optional[int] = None
    
    # `seq_level_k_best_ranked`:
    beta_decay: Optional[float] = 2.
    
    
    # ======== Other ========
    is_hpt: bool = False
    smart_load: bool = True
    force_reprocess_dataset: bool = False
    force_reprocess_k_best: bool = False
    eval_first_step: bool = False
    log_preds_to_wandb: bool = True
    log_raw_str: bool = False
    n_samples_per_wandb_logging_step: int = 8
    
    
    
    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        
        # Set defaults:
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if self.early_stopping_patience is None:
            self.early_stopping_patience = -1
        
        assert self.save_total_limit is None or self.save_total_limit >= 2, \
            "The `save_total_limit` must be at least 2, or None."
        
        self._validate_distillation_args()
    
    
    def _validate_distillation_args(self) -> None:
        """Validate the distillation arguments."""
        
        assert self.method_distil in AVAILABLE_KD_METHODS, \
            f"Invalid distillation method `{self.method_distil}`. Available methods: {AVAILABLE_KD_METHODS}."
        
        if self.method_distil == "word_level":
            assert self.temperature is not None, \
                "The `temperature` must be set for `word_level` distillation."
        if self.method_distil in ["seq_level_k_best_uniform", "seq_level_k_best_ranked"]:
            assert self.distillation_num_beams is not None, \
                "The `distillation_num_beams` must be set for sequence-level distillation."
        if self.method_distil in ["seq_level_k_best_uniform", "seq_level_k_best_ranked"]:
            assert self.distillation_num_beams is not None and self.distillation_num_beams > 0, \
                "The `distillation_num_beams` must be greater than 0 for sequence-level distillation."
        if self.method_distil == "seq_level_k_best_ranked":
            assert self.beta_decay is not None, \
                "The `beta_decay` must be set for `seq_level_k_best_ranked` distillation."
            assert self.beta_decay > 0, \
                "The `beta_decay` must be greater than 0 for `seq_level_k_best_ranked` distillation."
    
    
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
        
        return DistilConfig(**config_dict)
