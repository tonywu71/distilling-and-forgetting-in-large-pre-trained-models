import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from dataloader.dataset_for_evaluation.esb_dataset import ESBDataset


def main():
    """
    Cache a dataset from HuggingFace Datasets.
    """
    
    # Print environment variables:
    print("Environment variables:")
    print_envs()
    print("\n-----------------------\n")
    
    
    print("Loading ESB dataset...")
    
    esb_dataset = ESBDataset(load_diagnostic=True, subset=None)
    
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
