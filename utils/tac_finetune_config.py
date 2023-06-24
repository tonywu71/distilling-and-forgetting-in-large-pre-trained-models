from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml

from utils.finetune_config import FinetuneConfig


@dataclass
class TACFinetuneConfig(FinetuneConfig):
    gamma_tac: float = 0.5
    languages_to_preserve: List[str] = field(default_factory=list)


    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        super().__post_init__()
        assert self.languages_to_preserve, "The `languages_to_preserve` must not be empty."
        assert not self.zero_shot, "Zero-shot learning is not supported for TAC."
    
    @staticmethod
    def from_yaml(config_file: str) -> "TACFinetuneConfig":
        """Parse the YAML config file and return a TACFinetuneConfig object"""
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert types:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])
        
        # Fix paths:
        if config_dict["model_dir"] and not config_dict["model_dir"].endswith("/"):
            # The model_dir must end with a slash:
            config_dict["model_dir"] = config_dict["model_dir"] + "/"
        
        return TACFinetuneConfig(**config_dict)