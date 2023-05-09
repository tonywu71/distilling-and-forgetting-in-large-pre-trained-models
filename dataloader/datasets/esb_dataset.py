from typing import Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup


class ESBDataset(BaseDatasetGroup):
    """
    Class that regroups the End-to-end Speech Benchmark (ESB) datasets.
    See for more details:
    - https://arxiv.org/abs/2210.13352 
    - https://huggingface.co/datasets/esb/datasets
    - https://huggingface.co/datasets/esb/diagnostic-dataset
    """
    
    def __init__(self,
                 streaming: bool=False,
                 load_diagnostic: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.dataset_path = "esb/datasets" if not load_diagnostic else "esb/diagnostic-dataset"
        self.available_datasets = [
            "librispeech",
            "common_voice",
            "voxpopuli",
            "tedlium",
            "gigaspeech",
            "spgispeech",
            "earnings22",
            "ami"
        ]
        self.is_multilingual = False
        self.language = "english"
        self.load_diagnostic = load_diagnostic
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        if not self.load_diagnostic:  # If load default ESB dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    if dataset_name == "librispeech":
                        # Load the 2 test splits of LibriSpeech from the original HF dataset as
                        # `esb/datasets` does not provide the text annotations for the test set:
                        self.str2dataset["librispeech_clean"] = load_dataset(path="librispeech_asr",
                                                                             name="clean",
                                                                             split="test",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                        self.str2dataset["librispeech_other"] = load_dataset(path="librispeech_asr",
                                                                             name="other",
                                                                             split="test",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                    else:
                        # For all other datasets, load the validation splits:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                      name=dataset_name,
                                                                      split="validation",
                                                                      streaming=self.streaming,
                                                                      use_auth_token=True)
        
        else:  # If load diagnostic dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    if dataset_name == "librispeech":
                        # Load the 2 splits of LibriSpeech from the original HF test dataset
                        # because LibriSpeech is our main dataset of interest (used for fine-tuning):
                        self.str2dataset["librispeech_clean"] = load_dataset(path="librispeech_asr",
                                                                             name="clean",
                                                                             split="test",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                        self.str2dataset["librispeech_other"] = load_dataset(path="librispeech_asr",
                                                                             name="other",
                                                                             split="test",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                      name=dataset_name,
                                                                      split="clean",
                                                                      streaming=self.streaming,
                                                                      use_auth_token=True
                                                                      ).rename_column("norm_transcript", "text")