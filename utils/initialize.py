import os

def initialize_env():
    """
    Initialize the environment variables for HuggingFace and WandB.
    """
    
    # HuggingFace:
    os.environ["HF_HOME"] = "/home/tw581/rds/hpc-work/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/home/tw581/rds/hpc-work/cache/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/home/tw581/rds/hpc-work/cache/huggingface/datasets"
    os.environ["HF_MODULES_CACHE"] = "/home/tw581/rds/hpc-work/cache/huggingface/modules"
    
    # WandB:
    os.environ["WANDB_CACHE_DIR"] = "/home/tw581/rds/hpc-work/cache/wandb"
    
    return


def print_envs():
    list_envs = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_MODULES_CACHE", "WANDB_CACHE_DIR"]
    for env in list_envs:
        print(f"{env}: {os.environ[env]}")
    return
