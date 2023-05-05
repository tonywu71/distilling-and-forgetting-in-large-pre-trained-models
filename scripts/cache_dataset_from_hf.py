import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from datasets import load_dataset


def main(path: str=typer.Argument(..., help="Path to the dataset."),
         name: str=typer.Option(default=None, help="Name of the dataset."),
         split: str=typer.Option(default=None, help="Split of the dataset.")):
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    print(f"Loading dataset {name} from {path}...")
    
    if split:
        print(f"Split: {split}...")
    
    dataset = load_dataset(path=path, name=name, split=split)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
