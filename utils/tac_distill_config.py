from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml

from utils.distil_config import DistilConfig

@dataclass
class TACDistilConfig(DistilConfig):
    gamma_tac: float = 0.5
    languages_to_preserve: List[str] = []
    method_tac: Optional[str] = None


    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        
        super().__post_init__()
        assert self.languages_to_preserve, "The `languages_to_preserve` must not be empty."
    
    @staticmethod
    def from_yaml(config_file: str) -> "TACDistilConfig":
        """Parse the YAML config file and return a TACDistilConfig object"""
        assert Path(config_file).exists(), f"Config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert types:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])
        
        # Fix paths:
        if config_dict["model_dir"] and not config_dict["model_dir"].endswith("/"):
            # The model_dir must end with a slash:
            config_dict["model_dir"] = config_dict["model_dir"] + "/"
        
        return TACDistilConfig(**config_dict)
