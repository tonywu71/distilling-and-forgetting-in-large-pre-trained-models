from datasets import Dataset

from dataloader.dataset_loader import TRAIN_DATASET_NAME_TO_LOAD_FCT
from evaluation.eval_dataset_name_to_dataset_group import EVAL_DATASET_NAME_TO_DATASET_GROUP


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name in ["ami_100h_train", "ami_10h_train"]:
        dataset_dict = TRAIN_DATASET_NAME_TO_LOAD_FCT[dataset_name]()
        ds = dataset_dict["train"]
    elif dataset_name in ["ami_test", "ami_10h_test"]:
        dataset_name = dataset_name.replace("_test", "")
        ds_group = EVAL_DATASET_NAME_TO_DATASET_GROUP[dataset_name]()
        ds = ds_group.str2dataset["ami"]
        ds = ds.map(lambda x: {"text": x.lower()}, input_columns=["text"])
    elif dataset_name in ["librispeech_dummy", "ami_validation"]:
        ds_group = EVAL_DATASET_NAME_TO_DATASET_GROUP[dataset_name]()
        ds = ds_group.str2dataset[dataset_name]
        ds = ds.map(lambda x: {"text": x.lower()}, input_columns=["text"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")
    
    ds = ds.map(lambda x: {"text": x.lower()}, input_columns=["text"])
    
    return ds
