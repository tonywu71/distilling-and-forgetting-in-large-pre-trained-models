import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from datasets import load_dataset, DatasetDict


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
    dataset_dict["train"] = load_dataset("edinburghcstr/ami",
                                         name="ihm",
                                         split="train",
                                         cache_dir=cache_dir_ami)
    dataset_dict["validation"] = load_dataset("edinburghcstr/ami",
                                              name="ihm",
                                              split="validation",
                                              cache_dir=cache_dir_ami)
    dataset_dict["test"] = load_dataset("edinburghcstr/ami",
                                        name="ihm",
                                        split="test",
                                        cache_dir=cache_dir_ami)
    dataset_dict = DatasetDict(dataset_dict)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
