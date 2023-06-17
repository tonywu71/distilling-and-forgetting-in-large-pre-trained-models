"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

import os
from datasets import Dataset, DatasetDict, load_dataset


def remove_unnecessary_cols_for_librispeech(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Remove unnecessary columns from the LibriSpeech dataset."""
    return dataset.remove_columns(column_names=["file", "speaker_id", "chapter_id", "id"])


def load_librispeech_dummy() -> DatasetDict:
    """DEBUG ONLY. Load the LibriSpeech dummy dataset.
    Important note: Because the dummy dataset only has 1 split available, we will use it for train, eval and test splits."""
    
    cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
    if cache_dir_librispeech is None:
        print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_librispeech}`.")
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                                         name="clean",
                                         split="validation",
                                         cache_dir=cache_dir_librispeech)
    dataset_dict["validation"] = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                                              name="clean",
                                              split="validation",
                                              cache_dir=cache_dir_librispeech)
    dataset_dict["test"] = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                                        name="clean",
                                        split="validation",
                                        cache_dir=cache_dir_librispeech)
    dataset_dict = DatasetDict(dataset_dict)
    
    # Remove unnecessary columns from the dataset:
    dataset_dict = remove_unnecessary_cols_for_librispeech(dataset_dict)
    
    # Column renaming is not necessary here because the dummy dataset already has the correct column name.
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict  # type: ignore


def load_librispeech(train_split: str="train.100") -> DatasetDict:
    """Load the train/eval/test splits of the LibriSpeech dataset."""
    
    cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
    if cache_dir_librispeech is None:
        print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_librispeech}`.")
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("librispeech_asr",
                                         name="clean",
                                         split=train_split,
                                         cache_dir=cache_dir_librispeech)
    dataset_dict["validation"] = load_dataset("librispeech_asr",
                                              name="clean",
                                              split="validation",
                                              cache_dir=cache_dir_librispeech)
    dataset_dict["test"] = load_dataset("librispeech_asr",
                                        name="clean",
                                        split="test",
                                        cache_dir=cache_dir_librispeech)
    dataset_dict = DatasetDict(dataset_dict)
    
    # Remove unnecessary columns from the dataset:
    dataset_dict = remove_unnecessary_cols_for_librispeech(dataset_dict)
    
    # Column renaming is not necessary here because the LibriSpeech dataset already has the correct column name.
    # dataset_dict = rename_label_col(dataset_dict,
    #                                 old_label_col="text",
    #                                 new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict  # type: ignore
