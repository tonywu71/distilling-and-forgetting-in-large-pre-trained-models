from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

from utils.finetune_config import FinetuneConfig


@dataclass
class EWCFinetuneConfig(FinetuneConfig):
    dirpath_ewc: Optional[str] = None
    lambda_ewc: float = 0.1
    
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.regularization_method = "ewc"
        assert self.dirpath_ewc is not None, "`dirpath_ewc` must be specified."
        assert Path(self.dirpath_ewc).exists(), f"`dirpath_ewc` does not exist: {self.dirpath_ewc}"
    
    
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
