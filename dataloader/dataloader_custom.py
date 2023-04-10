"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_COL`.
"""

from datasets import DatasetDict, load_dataset
from utils.constants import DEFAULT_LABEL_COL


def rename_label_col(dataset_dict: DatasetDict,
                     old_label_col: str,
                     new_label_col: str=DEFAULT_LABEL_COL) -> DatasetDict:
    """Rename the label column in dataset_dict."""
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].rename_column(old_label_col, new_label_col)
    return dataset_dict


def load_librispeech(**kwargs) -> DatasetDict:
    """Load the LibriSpeech dataset."""
    old_label_col = "text"
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("librispeech_asr", name="clean", split="train.100")
    dataset_dict["test"] = load_dataset("librispeech_asr", name="clean", split="test")
    dataset_dict = DatasetDict(dataset_dict)
    
    dataset_dict = rename_label_col(dataset_dict, old_label_col=old_label_col)
    
    return dataset_dict
