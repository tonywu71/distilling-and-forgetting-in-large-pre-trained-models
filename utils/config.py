from dataclasses import dataclass
import yaml


@dataclass
class Config:
    """Config class for the Whisper experiments"""
    lang_name: str
    lang_id: str
    pretrained_model_name_or_path: str
    model_dir: str


def parse_yaml_config(config_file: str) -> Config:
    """Parse the YAML config file and return a Config object"""
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
