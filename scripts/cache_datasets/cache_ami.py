import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from datasets import load_dataset, concatenate_datasets

from dataloader.dataloader_custom.dataloader_ami import LIST_SUBSETS_AMI


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
    
    dict_ami_per_split = {
        "train": [],
        "validation": [],
        "test": []
    }
    dataset_dict = {}
    
    for split in dict_ami_per_split:
        for subset in LIST_SUBSETS_AMI:
            print("Loading subset `{}` for split `{}`...".format(subset, split))
            dict_ami_per_split[split].append(load_dataset("edinburghcstr/ami",
                                                          name=subset,
                                                          split=split,
                                                          cache_dir=cache_dir_ami))
    
    for split, list_ds in dict_ami_per_split.items():
        dataset_dict[split] = concatenate_datasets(list_ds)  # type: ignore
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
