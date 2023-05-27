"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

import os
from datasets import DatasetDict, load_dataset, concatenate_datasets


LIST_SUBSETS_AMI = ["ihm", "sdm"]


def load_ami() -> DatasetDict:
    """Load the train/eval/test splits of the LibriSpeech dataset."""
    
    cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
    if cache_dir_ami is None:
        print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_ami}`.")
    
    dict_ami_per_split = {
        "train": [],
        "validation": [],
        "test": []
    }

    dataset_dict = {}
    
    for split, list_ds in dict_ami_per_split.items():
        for subset in LIST_SUBSETS_AMI:
            dict_ami_per_split[split].append(load_dataset("edinburghcstr/ami",
                                                          name=subset,
                                                          split=split,
                                                          cache_dir=cache_dir_ami))
    
    for split, list_ds in dict_ami_per_split.items():
        dataset_dict[split] = concatenate_datasets(list_ds)  # type: ignore
    
    dataset_dict = DatasetDict(dataset_dict)
    
    # Column renaming is not necessary here because the AMI dataset already has the correct column name.    
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict
