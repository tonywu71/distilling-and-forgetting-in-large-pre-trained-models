import os

from utils.env_config import load_yaml_env_config
from utils.constants import DEFAULT_ENV_CONFIG_FILEPATH


def initialize_env():
    """
    Initialize the environment variables for HuggingFace and WandB.
    
    By default, the environment variables are loaded from the file `configs/env_config.yaml`.
    One can also specify the path to the config file using the environment variable `ENV_CONFIG_FILEPATH`.
    """
    
    env_config_filepath = os.environ.get("ENV_CONFIG_FILEPATH", DEFAULT_ENV_CONFIG_FILEPATH)
    env_config = load_yaml_env_config(env_config_filepath)
    
    # HuggingFace:
    os.environ["HF_HOME"] = env_config.HF_HOME
    
    os.environ["TRANSFORMERS_CACHE"] = env_config.TRANSFORMERS_CACHE
    os.environ["HF_DATASETS_CACHE"] = env_config.HF_DATASETS_CACHE
    os.environ["HF_MODULES_CACHE"] = env_config.HF_MODULES_CACHE
    
    for var in ["CACHE_DIR_LIBRISPEECH", "CACHE_DIR_AMI", "CACHE_DIR_ESB",
                "CACHE_DIR_ESB_DIAGNOSTIC", "CACHE_DIR_MLS"]:
        if getattr(env_config, var) is not None:
            os.environ[var] = getattr(env_config, var)
    
    
    # WandB:
    os.environ["WANDB_PROJECT_TRAINING"] = env_config.WANDB_PROJECT_TRAINING
    os.environ["WANDB_PROJECT_EVALUATION"] = env_config.WANDB_PROJECT_EVALUATION
    os.environ["WANDB_PROJECT_OTHERS"] = env_config.WANDB_PROJECT_OTHERS
    os.environ["WANDB_CACHE_DIR"] = env_config.WANDB_CACHE_DIR
    
    # Other:
    os.environ["PREPROCESSED_DATASETS_DIR"] = env_config.PREPROCESSED_DATASETS_DIR
    os.environ["K_BEAM_SEARCH_CACHE_DIR"] = env_config.K_BEAM_SEARCH_CACHE_DIR
    
    return


def print_envs():
    list_envs = [
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_MODULES_CACHE",
        "CACHE_DIR_LIBRISPEECH",
        "CACHE_DIR_AMI",
        "CACHE_DIR_ESB",
        "CACHE_DIR_ESB_DIAGNOSTIC",
        "CACHE_DIR_MLS",
        "WANDB_PROJECT_TRAINING",
        "WANDB_PROJECT_EVALUATION",
        "WANDB_CACHE_DIR",
        "PREPROCESSED_DATASETS_DIR",
        "K_BEAM_SEARCH_CACHE_DIR"
    ]
    
    for env in list_envs:
        print(f"{env}: {os.environ.get(env, None)}")
    
    return
