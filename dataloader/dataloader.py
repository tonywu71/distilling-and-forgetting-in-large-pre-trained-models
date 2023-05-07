from functools import partial
from typing import Iterable
from datasets import DatasetDict

from dataloader.dataloader_custom import load_librispeech, load_librispeech_dummy
from utils.constants import DEFAULT_LABEL_STR_COL


STR_TO_LOAD_FCT = {
    "librispeech_clean_100h": partial(load_librispeech, train_split="train.100"),
    "librispeech_clean_360h": partial(load_librispeech, train_split="train.360"),
    "librispeech_dummy": load_librispeech_dummy
}


def load_dataset_dict(dataset_name: str, **kwargs) -> DatasetDict:
    """Load the dataset dictionary."""
    if dataset_name in STR_TO_LOAD_FCT:
        dataset_dict = STR_TO_LOAD_FCT[dataset_name](**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    for split in ["train", "val", "test"]:
        assert split in dataset_dict, f"Split {split} not found in dataset {dataset_name}"
    
    return dataset_dict


def gen_from_dataset(dataset) -> Iterable[dict]:
    """Yield the audio and reference from the dataset."""
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item[DEFAULT_LABEL_STR_COL]}
