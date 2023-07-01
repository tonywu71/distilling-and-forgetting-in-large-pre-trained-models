import os
from typing import Optional, List
from datasets import load_dataset

from dataloader.dataset_for_training.dataset_loader_ami import remove_unnecessary_cols_for_ami
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup


class AMITestSet(BaseDatasetGroup):
    """
    Util DatasetGroup to eval the vanilla Whisper model on the AMI set.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 is_ami_10h: bool = False,
                 subset: Optional[List[str]]=None) -> None:
        super().__init__(streaming=streaming, subset=subset)
        self.is_ami_10h = is_ami_10h
        
        if self.is_ami_10h:
            assert not self.streaming, "Streaming is not supported for the 10h AMI set."
        
        # Set the abstract class attributes:
        self.available_datasets = [
            "ami_test"
        ]
        self.is_multilingual = False
        self.language = "english"
        
        self.post_init()
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "ami_test": load_dataset("edinburghcstr/ami",
                                     name="ihm",
                                     split="test" if not self.is_ami_10h else "test[:10%]",
                                     cache_dir=self.cache_dir_ami,
                                     streaming=self.streaming)
        }
        
        self.str2dataset["ami_test"] = remove_unnecessary_cols_for_ami(self.str2dataset["ami_test"])


    def _load_cache_dir_from_env_var(self) -> None:
        self.cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
        if self.cache_dir_ami is None:
            print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_ami}`.")
        return
