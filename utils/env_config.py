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
    
    TRANSFORMERS_CACHE: str
    HF_DATASETS_CACHE: str
    HF_MODULES_CACHE: str
    
    CACHE_DIR_LIBRISPEECH: Optional[str]
    CACHE_DIR_AMI: Optional[str]
    CACHE_DIR_ESB: Optional[str]
    CACHE_DIR_ESB_DIAGNOSTIC: Optional[str]
    CACHE_DIR_MLS: Optional[str]
    
    # ------ Weight&Biases ------
    WANDB_PROJECT_TRAINING: str
    WANDB_PROJECT_EVALUATION: str
    WANDB_CACHE_DIR: str
    
    # ------ Other ------
    PREPROCESSED_DATASETS_DIR: str
    K_BEAM_SEARCH_CACHE_DIR: str


def load_yaml_env_config(config_file: str) -> EnvConfig:
    """Parse the YAML config file and return an EnvConfig object"""
    assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return EnvConfig(**config_dict)
