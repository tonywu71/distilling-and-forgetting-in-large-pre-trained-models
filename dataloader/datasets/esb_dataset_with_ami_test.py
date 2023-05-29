import os

from typing import Optional, List
from datasets import load_dataset, concatenate_datasets

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from dataloader.dataloader_custom.dataloader_ami import LIST_SUBSETS_AMI


class ESBDatasetWithAMITest(BaseDatasetGroup):
    """
    ESB dataset group with the LibriSpeech dataset replaced with
    the test split of the original AMI dataset.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 load_diagnostic: bool=True,
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
        
        
        # Retrieve custom `cache_dir` filepath if set:
        if self.load_diagnostic:
            self.cache_dir_esb = os.environ.get("CACHE_DIR_ESB_DIAGNOSTIC", None)
            if self.cache_dir_esb is None:
                print("WARNING: `CACHE_DIR_ESB_DIAGNOSTIC` environment variable not set. Using default cache directory.")
            else:
                print(f"Using cache directory: `{self.cache_dir_esb}`.")
        else:
            self.cache_dir_esb = os.environ.get("CACHE_DIR_ESB", None)
            if self.cache_dir_esb is None:
                print("WARNING: `CACHE_DIR_ESB` environment variable not set. Using default cache directory.")
            else:
                print(f"Using cache directory: `{self.cache_dir_esb}`.")
        
        self.cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
        if self.cache_dir_ami is None:
            print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_ami}`.")
        
        
        self.dataset_name_to_cache_dir = {
            "librispeech": self.cache_dir_esb,
            "common_voice": self.cache_dir_esb,
            "voxpopuli": self.cache_dir_esb,
            "tedlium": self.cache_dir_esb,
            "gigaspeech": self.cache_dir_esb,
            "spgispeech": self.cache_dir_esb,
            "earnings22": self.cache_dir_esb,
            "ami": self.cache_dir_ami,
        }
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        if not self.load_diagnostic:  # If `load_diagnostic` default ESB dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    
                    if dataset_name == "ami":
                        list_ds = []
                        for subset_ami in LIST_SUBSETS_AMI:
                            list_ds.append(load_dataset(path="edinburghcstr/ami",
                                                        name=subset_ami,
                                                        split="test",
                                                        streaming=self.streaming,
                                                        cache_dir=self.cache_dir_ami))
                        self.str2dataset["ami"] =  concatenate_datasets(list_ds)
                        
                    else:
                        # For all other datasets, load the validation splits:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                      name=dataset_name,
                                                                      split="validation",
                                                                      cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                      streaming=self.streaming,
                                                                      use_auth_token=True)
        
        else:  # If load diagnostic dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    
                    if dataset_name == "ami":
                        list_ds = []
                        for subset_ami in LIST_SUBSETS_AMI:
                            list_ds.append(load_dataset("edinburghcstr/ami",
                                                        name=subset_ami,
                                                        split="test",
                                                        streaming=self.streaming,
                                                        cache_dir=self.cache_dir_ami))
                        self.str2dataset["ami"] =  concatenate_datasets(list_ds)
                    
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                      name=dataset_name,
                                                                      split="clean",
                                                                      cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                      streaming=self.streaming,
                                                                      use_auth_token=True
                                                                      ).rename_column("norm_transcript", "text")
