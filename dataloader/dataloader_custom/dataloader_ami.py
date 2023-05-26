"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

import os
from datasets import DatasetDict, load_dataset


def load_ami() -> DatasetDict:
    """Load the train/eval/test splits of the LibriSpeech dataset."""
    
    cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
    if cache_dir_ami is None:
        print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: {cache_dir_ami}")
    
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
    dataset_dict = DatasetDict(dataset_dict)
    
    # Column renaming is not necessary here because the AMI dataset already has the correct column name.    
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict
