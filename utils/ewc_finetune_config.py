from dataclasses import dataclass
from pathlib import Path
import yaml

from utils.finetune_config import FinetuneConfig


@dataclass
class EWCFinetuneConfig(FinetuneConfig):
    lamda_ewc: float = 0.1    
    
    @staticmethod
    def from_yaml(config_file: str) -> "EWCFinetuneConfig":
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
        
        return EWCFinetuneConfig(**config_dict)
