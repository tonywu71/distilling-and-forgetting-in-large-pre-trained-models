import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from dataloader.datasets.fab_dataset import FABDataset


def main():
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    print("Loading FAB dataset...")
    
    fab_dataset = FABDataset(subset=None)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
