from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class EnvConfig:
    """
    Environment config class
    """
    # ------ Huggingface ------
    HF_HOME: str
    
    HUGGING_FACE_HUB_TOKEN: Optional[str]
    
    TRANSFORMERS_CACHE: str
    HF_DATASETS_CACHE: str
    HF_MODULES_CACHE: str
    
    # ------ Weight&Biases ------
    WANDB_PROJECT: str
    WANDB_CACHE_DIR: str
    
    # ------ Other ------
    PREPROCESSED_DATASETS_DIR: str


def load_yaml_env_config(config_file: str) -> EnvConfig:
    """Parse the YAML config file and return an EnvConfig object"""
    assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return EnvConfig(**config_dict)
