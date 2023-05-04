"""
Note: For each dataset, the label column must be renamed to the value stored in `DEFAULT_LABEL_STR_COL`.
"""

from datasets import DatasetDict, load_dataset
from utils.constants import DEFAULT_LABEL_STR_COL


def rename_label_col(dataset_dict: DatasetDict,
                     old_label_col: str,
                     new_label_col: str) -> DatasetDict:
    """Rename the label column in dataset_dict."""
    if old_label_col == new_label_col:  # otherwise, we will get an error
        return dataset_dict
    
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].rename_column(old_label_col, new_label_col)
    return dataset_dict


def load_librispeech_dummy() -> DatasetDict:
    """DEBUG ONLY. Load the LibriSpeech dummy dataset.
    Important note: Because the dummy dataset only has 1 split available, we will use it for train, eval and test splits."""
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("hf-internal-testing/librispeech_asr_dummy", name="clean", split="validation")
    dataset_dict["val"] = load_dataset("hf-internal-testing/librispeech_asr_dummy", name="clean", split="validation")
    dataset_dict["test"] = load_dataset("hf-internal-testing/librispeech_asr_dummy", name="clean", split="validation")
    dataset_dict = DatasetDict(dataset_dict)
    
    dataset_dict = rename_label_col(dataset_dict,
                                    old_label_col="text",
                                    new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict


def load_librispeech(train_split: str="train.100") -> DatasetDict:
    """Load the train/eval/test splits of the LibriSpeech dataset."""
    
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("librispeech_asr", name="clean", split=train_split)
    dataset_dict["val"] = load_dataset("librispeech_asr", name="clean", split="validation")
    dataset_dict["test"] = load_dataset("librispeech_asr", name="clean", split="test")
    dataset_dict = DatasetDict(dataset_dict)
    
    dataset_dict = rename_label_col(dataset_dict,
                                    old_label_col="text",
                                    new_label_col=DEFAULT_LABEL_STR_COL)
    
    return dataset_dict
