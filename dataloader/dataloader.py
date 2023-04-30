from functools import partial
from datasets import DatasetDict
from dataloader.dataloader_custom import load_librispeech, load_librispeech_dummy


STR_TO_LOAD_FCT = {
    "librispeech_100h": partial(load_librispeech, train_split="train.100"),
    "librispeech_360h": partial(load_librispeech, train_split="train.360"),
    "librispeech_dummy": load_librispeech_dummy
}


def load_dataset_dict(dataset_name: str, **kwargs) -> DatasetDict:
    """Load the dataset dictionary."""
    if dataset_name in STR_TO_LOAD_FCT:
        dataset_dict = STR_TO_LOAD_FCT[dataset_name](**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset_dict
