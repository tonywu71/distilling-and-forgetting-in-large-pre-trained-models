from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DistilConfig:
    """
    Config class for distillation experiments.
    """
    experiment_name: str
    lang_name: str
    lang_id: str
    task: str
    teacher_model_name_or_path: str
    student_model_name_or_path: str
    finetuned_from: str
    model_dir: str
    freeze_encoder: bool
    freeze_decoder: bool
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
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
    log_preds_to_wandb: bool = True
    
    
    @staticmethod
    def from_yaml(config_file: str) -> "DistilConfig":
        """Parse the YAML config file and return a DistillConfig object"""
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Sanity checks:
        assert config_dict["save_total_limit"] is None or config_dict["save_total_limit"] >= 2, \
            "The save_total_limit must be at least 2, or None."
        
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
