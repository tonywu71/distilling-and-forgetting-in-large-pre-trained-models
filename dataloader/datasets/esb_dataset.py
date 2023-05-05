from typing import Callable, Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from utils.constants import DEFAULT_NUM_PROC


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
        self.load_diagnostic = load_diagnostic
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        if not self.load_diagnostic:  # If load default ESB dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    if dataset_name == "librispeech":
                        self.str2dataset["librispeech_clean"] = load_dataset(path=self.dataset_path,
                                                                             name=dataset_name,
                                                                             split="test.clean",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                        self.str2dataset["librispeech_other"] = load_dataset(path=self.dataset_path,
                                                                             name=dataset_name,
                                                                             split="test.other",
                                                                             streaming=self.streaming,
                                                                             use_auth_token=True)
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                      name=dataset_name,
                                                                      split="test",
                                                                      streaming=self.streaming,
                                                                      use_auth_token=True)
        
        else:  # If load diagnostic dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                name=dataset_name,
                                                                split="clean",
                                                                streaming=self.streaming,
                                                                use_auth_token=True)
    
        
    def preprocess_datasets(self,
                            normalize_fct: Optional[Callable]=None) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        
        # Loop over all the datasets in the ESB benchmark:
        for dataset_name, dataset in self.str2dataset.items():
            if self.load_diagnostic:
                dataset = dataset.rename_column("norm_transcript", "text")
            
            # Normalize references (especially important for Whisper):
            if normalize_fct:
                dataset = dataset.map(normalize_fct, num_proc=DEFAULT_NUM_PROC)
            # Update dataset:
            self.str2dataset[dataset_name] = dataset

        self.preprocessed = True
        return
