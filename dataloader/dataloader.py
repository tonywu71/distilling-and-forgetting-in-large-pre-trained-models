from dataloader.dataloader_custom import *


STR_TO_LOAD_FCT = {
    "librispeech": load_librispeech
}


def load_dataset_dict(dataset_name: str, **kwargs) -> dict:
    """Load the dataset dictionary."""
    if dataset_name in STR_TO_LOAD_FCT:
        dataset_dict = STR_TO_LOAD_FCT[dataset_name](**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset_dict


def shuffle_dataset_dict(dataset_dict: dict, buffer_size: int=1000) -> dict:
    """Shuffle the dataset dictionary."""
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].shuffle(buffer_size=buffer_size).with_format("torch")
    return dataset_dict


def convert_dataset_dict_to_torch(dataset_dict: dict) -> dict:
    """Convert the dataset dictionary to PyTorch format."""
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].with_format("torch")
    return dataset_dict
