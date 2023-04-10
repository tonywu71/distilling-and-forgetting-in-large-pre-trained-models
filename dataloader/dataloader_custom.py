"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_COL`.
"""

from datasets import DatasetDict, load_dataset
from utils.constants import DEFAULT_LABEL_COL


def load_librispeech(**kwargs) -> DatasetDict:
    """Load the LibriSpeech dataset."""
    OLD_LABEL_COL = "text"
    
    dataset_dict = {}
    
    dataset_dict["train"] = load_dataset("librispeech_asr", name="clean", split="train.100")
    dataset_dict["test"] = load_dataset("librispeech_asr", name="clean", split="test")
    
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].rename_column(OLD_LABEL_COL, DEFAULT_LABEL_COL)
    
    return DatasetDict(dataset_dict)
