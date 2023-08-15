from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

from utils.finetune_config import FinetuneConfig


@dataclass
class TACFinetuneConfig(FinetuneConfig):
    gamma_tac: float = 0.1
    languages_to_preserve: List[str] = field(default_factory=list)
    task_tac: str = "transcribe"
    use_kl: bool = False
    temperature: float = 1.0


    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        super().__post_init__()
        self.regularization_method = "tac"
        assert self.languages_to_preserve, "The `languages_to_preserve` must not be empty."
    
    
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
