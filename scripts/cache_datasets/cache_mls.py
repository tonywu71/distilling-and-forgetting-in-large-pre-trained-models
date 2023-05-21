import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from dataloader.datasets.mls_dataset import MLSDataset


def main():
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    # Print environment variables:
    print("Environment variables:")
    print_envs()
    print("\n-----------------------\n")
    
    
    print("Loading MLS dataset...")
    
    mls_dataset = MLSDataset(subset=None)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
