from typing import Optional, List

import os

from datasets import load_dataset, concatenate_datasets

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from dataloader.dataloader_custom.dataloader_ami import LIST_SUBSETS_AMI


class AMITestSet(BaseDatasetGroup):
    """
    Util DatasetGroup to eval the vanilla Whisper model on the AMI set.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.available_datasets = [
            "ami_test"
        ]
        
        self.is_multilingual = False
        self.language = "english"
        
        # Retrieve custom `cache_dir` filepath if set:
        self.cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
        if self.cache_dir_ami is None:
            print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_ami}`.")
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        list_ds = []
        for subset in LIST_SUBSETS_AMI:
            list_ds.append(load_dataset("edinburghcstr/ami",
                                        name=subset,
                                        split="test",
                                        streaming=self.streaming,
                                        cache_dir=self.cache_dir_ami))
        
        self.str2dataset = {
            "ami_test": concatenate_datasets(list_ds)
        }


class AMITestSet1H(AMITestSet):
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        super().__init__(streaming=streaming, subset=subset)
    
    def _prepare_str2dataset(self) -> None:
        list_ds = []
        for subset in LIST_SUBSETS_AMI:
            list_ds.append(load_dataset("edinburghcstr/ami",
                                        name=subset,
                                        split="test[:10%]",  # 10% of the 10h test set = 1h
                                        streaming=self.streaming,
                                        cache_dir=self.cache_dir_ami))
        
        self.str2dataset = {
            "ami_test": concatenate_datasets(list_ds)
        }
