import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from datasets import load_dataset


def main():
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    # Print environment variables:
    print("Environment variables:")
    print_envs()
    print("\n-----------------------\n")
    
    
    cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
    if cache_dir_ami is None:
        print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_ami}`.")
    
    
    print("Loading MLS dataset...")
    
    dataset_dict = {}
    
    dataset_dict["train"] = load_dataset("esb/datasets",
                                         name="ami",
                                         split="train",
                                         cache_dir=cache_dir_ami)
    dataset_dict["val"] = load_dataset("esb/datasets",
                                       name="ami",
                                       split="validation",
                                       cache_dir=cache_dir_ami)
    dataset_dict["test"] = load_dataset("esb/datasets",
                                        name="ami",
                                        split="test",
                                        cache_dir=cache_dir_ami)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
