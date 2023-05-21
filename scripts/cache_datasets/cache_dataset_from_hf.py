import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from datasets import load_dataset


def main(path: str=typer.Argument(..., help="Path to the dataset."),
         name: str=typer.Option(default=None, help="Name of the dataset."),
         split: str=typer.Option(default=None, help="Split of the dataset.")):
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    # Print environment variables:
    print("Environment variables:")
    print_envs()
    print("\n-----------------------\n")
    
    
    print(f"Loading dataset {name} from {path}...")
    
    if split:
        print(f"Split: {split}...")
    
    dataset = load_dataset(path=path, name=name, split=split, use_auth_token=True)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
