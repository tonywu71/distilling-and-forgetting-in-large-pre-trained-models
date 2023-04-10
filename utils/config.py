from dataclasses import dataclass
import yaml


@dataclass
class Config:
    """Config class for the Whisper experiments"""
    experiment_name: str
    lang_name: str
    lang_id: str
    pretrained_model_name_or_path: str
    model_dir: str
    batch_size: int
    data_augmentation: bool
    dataset_dir: str
    optim: str
    learning_rate: float
    warmup_steps: int
    num_train_epochs: int


def load_yaml_config(config_file: str) -> Config:
    """Parse the YAML config file and return a Config object"""
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Convert types:
    config_dict["learning_rate"] = float(config_dict["learning_rate"])
    
    return Config(**config_dict)
