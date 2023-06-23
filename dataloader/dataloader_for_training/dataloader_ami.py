"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

import os
from datasets import Dataset, DatasetDict, load_dataset


def remove_unnecessary_cols_for_ami(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Remove unnecessary columns from the AMI dataset."""
    return dataset.remove_columns(column_names=["meeting_id", "audio_id", "begin_time", "end_time", "microphone_id", "speaker_id"])


def load_ami_100h() -> DatasetDict:
    """Load the train/eval/test splits of the AMI dataset."""
    
    cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
    if cache_dir_ami is None:
        print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_ami}`.")
    
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
    
    # Remove unnecessary columns from the dataset:
    dataset_dict = remove_unnecessary_cols_for_ami(dataset_dict)
    
    # Column renaming is not necessary here because the AMI dataset already has the correct column name.    
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict


def load_ami_10h() -> DatasetDict:
    """Load the train/eval/test splits of the AMI dataset.
    Only 10% of the 100h-dataset is loaded, i.e. 10h of audio data."""
    
    cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
    if cache_dir_ami is None:
        print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_ami}`.")
    
    # We load the 10h-dataset by sampling the first 10% splits of the 100h-dataset.
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("edinburghcstr/ami",
                                         name="ihm",
                                         split="train[:10%]",
                                         cache_dir=cache_dir_ami)
    dataset_dict["validation"] = load_dataset("edinburghcstr/ami",
                                              name="ihm",
                                              split="validation[:10%]",
                                              cache_dir=cache_dir_ami)
    dataset_dict["test"] = load_dataset("edinburghcstr/ami",
                                        name="ihm",
                                        split="test[:10%]",
                                        cache_dir=cache_dir_ami)
    dataset_dict = DatasetDict(dataset_dict)
    
    # Remove unnecessary columns from the dataset:
    dataset_dict = remove_unnecessary_cols_for_ami(dataset_dict)
    
    # Column renaming is not necessary here because the AMI dataset already has the correct column name.    
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict
