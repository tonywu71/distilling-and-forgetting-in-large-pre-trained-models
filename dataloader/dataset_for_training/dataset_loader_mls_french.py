"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

import os
from datasets import Dataset, DatasetDict, load_dataset


def remove_unnecessary_cols_for_mls(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Remove unnecessary columns from the AMI dataset."""
    return dataset.remove_columns(column_names=["file", "speaker_id", "chapter_id", "id"])


def load_mls(language: str, is_train_diagnostic: bool = False) -> DatasetDict:
    """
    Load the train/validation splits of the MLS dataset for a given language.
    If `is_train_diagnostic` is True, then the train split will be a small subset of the original train split.
    """
    
    cache_dir_mls = os.environ.get("CACHE_DIR_MLS", None)
    if cache_dir_mls is None:
        print("WARNING: `CACHE_DIR_MLS` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_mls}`.")
    
    available_languages = [
        "dutch",
        "english",
        "french",
        "german",
        "italian",
        "polish",
        "portuguese",
        "spanish"
    ]
    assert language in available_languages, f"Language `{language}` not supported. Available languages: {available_languages}."
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset(path="facebook/multilingual_librispeech",
                                         name=language,
                                         split="train[:20%]" if is_train_diagnostic else "train",
                                         cache_dir=cache_dir_mls)
    dataset_dict["validation"] = load_dataset(path="facebook/multilingual_librispeech",
                                              name=language,
                                              split="validation",
                                              cache_dir=cache_dir_mls)
    dataset_dict = DatasetDict(dataset_dict)
    
    # Remove unnecessary columns from the dataset:
    dataset_dict = remove_unnecessary_cols_for_mls(dataset_dict)
    
    # Column renaming is not necessary here because the MLS dataset already has the correct column name.    
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict
