import os

def initialize_env():
    """
    Set the HF_DATASETS_CACHE and WANDB_CACHE_DIR to the cache directory.
    """
    os.environ["HF_DATASETS_CACHE"] = "/rds/user/tw581/rds/cache/huggingface"
    os.environ["WANDB_CACHE_DIR"] = "/rds/user/tw581/rds/cache/wandb"
    return
