import os

from utils.env_config import load_yaml_env_config
from utils.constants import DEFAULT_ENV_CONFIG_FILEPATH, LIST_ENV_VARS


def initialize_env() -> None:
    """
    Initialize the environment variables for HuggingFace and WandB.
    
    By default, the environment variables are loaded from the file `configs/env_config.yaml`.
    One can also specify the path to the config file using the environment variable `ENV_CONFIG_FILEPATH`.
    """
    env_config_filepath = os.environ.get("ENV_CONFIG_FILEPATH", DEFAULT_ENV_CONFIG_FILEPATH)
    env_config = load_yaml_env_config(env_config_filepath)
    
    for var in LIST_ENV_VARS:
        if getattr(env_config, var) is not None:
            os.environ[var] = getattr(env_config, var)
    
    return


def print_envs() -> None:
    for env in LIST_ENV_VARS:
        print(f"{env}: {os.environ.get(env, None)}")
    return
