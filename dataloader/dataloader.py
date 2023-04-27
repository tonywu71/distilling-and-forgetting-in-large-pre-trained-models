from datasets import DatasetDict
from dataloader.dataloader_custom import load_librispeech, load_librispeech_dummy


STR_TO_LOAD_FCT = {
    "librispeech": load_librispeech,
    "librispeech_dummy": load_librispeech_dummy
}


def load_dataset_dict(dataset_name: str, **kwargs) -> DatasetDict:
    """Load the dataset dictionary."""
    if dataset_name in STR_TO_LOAD_FCT:
        dataset_dict = STR_TO_LOAD_FCT[dataset_name](**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset_dict


def shuffle_dataset_dict(dataset_dict: DatasetDict) -> DatasetDict:
    """Shuffle the dataset dictionary."""
    dataset_dict["train"] = dataset_dict["train"].shuffle()
    return dataset_dict


def convert_dataset_dict_to_torch(dataset_dict: DatasetDict) -> DatasetDict:
    """Convert the dataset dictionary to PyTorch format."""
    dataset_dict = dataset_dict.with_format("torch")
    return dataset_dict
