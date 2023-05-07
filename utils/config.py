from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    """
    Config class for the Whisper experiments
    """
    experiment_name: str
    lang_name: Optional[str]
    task: str
    pretrained_model_name_or_path: str
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


def load_yaml_config(config_file: str) -> Config:
    """Parse the YAML config file and return a Config object"""
    assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Convert types:
    config_dict["learning_rate"] = float(config_dict["learning_rate"])
    
    # Set defaults:
    if config_dict["early_stopping_patience"] is None:
        config_dict["early_stopping_patience"] = -1
    
    return Config(**config_dict)
