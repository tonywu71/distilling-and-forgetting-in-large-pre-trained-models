from functools import partial
from typing import Iterable
from datasets import DatasetDict

from dataloader.dataloader_for_training.dataloader_librispeech import load_librispeech, load_librispeech_dummy
from dataloader.dataloader_for_training.dataloader_ami import load_ami_100h, load_ami_10h
from dataloader.dataloader_for_training.dataloader_mls_french import load_mls_subset_train
from utils.constants import DEFAULT_LABEL_STR_COL


DATASET_NAME_TO_LOAD_FCT = {
    "librispeech_clean_100h": partial(load_librispeech, train_split="train.100"),
    "librispeech_clean_360h": partial(load_librispeech, train_split="train.360"),
    "librispeech_dummy": load_librispeech_dummy,
    "ami_100h": load_ami_100h,
    "ami_10h": load_ami_10h,
    "mls_french": partial(load_mls_subset_train, language="french"),
}


def load_dataset_dict(dataset_name: str, **kwargs) -> DatasetDict:
    """Load the dataset dictionary."""
    if dataset_name in DATASET_NAME_TO_LOAD_FCT:
        dataset_dict = DATASET_NAME_TO_LOAD_FCT[dataset_name](**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset_dict


def gen_from_dataset(dataset) -> Iterable[dict]:
    """Yield the audio and reference from the dataset."""
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item[DEFAULT_LABEL_STR_COL]}
