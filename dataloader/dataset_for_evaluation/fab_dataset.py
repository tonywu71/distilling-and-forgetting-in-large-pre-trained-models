from typing import Optional, List
import os
from toolz import dicttoolz
from datasets import load_dataset
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.dataset_for_training.dataset_loader_librispeech import remove_unnecessary_cols_for_librispeech
from dataloader.dataset_for_training.dataset_loader_ami import remove_unnecessary_cols_for_ami



class FABDataset(BaseDatasetGroup):
    """
    Class that regroups a few datasets as part of the Forgetting Assessment Benchmark (FAB).
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        super().__init__(streaming=streaming, subset=subset)
        
        # Set the abstract class attributes:
        self.available_datasets = [
            "librispeech_en_clean",
            "ami_validation",
            "ami",
            "tedlium",
            "librispeech_fr"
        ]
        self.is_multilingual = True
        self.ds_name_to_lang = {
            "librispeech_en_clean": "en",
            "ami_validation": "en",
            "ami": "en",
            "tedlium": "en",
            "librispeech_fr": "fr"
        }
        
        self._load_cache_dir_from_env_var()
        
        self.dataset_name_to_cache_dir = {
            "librispeech_en_clean": self.cache_dir_librispeech,
            "ami_validation": self.cache_dir_ami,
            "ami": self.cache_dir_ami,
            "tedlium": self.cache_dir_esb,
            "librispeech_fr": self.cache_dir_mls
        }
        
        self.post_init()
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_en_clean": load_dataset(path="librispeech_asr",
                                                 name="clean",
                                                 split="test",
                                                 cache_dir=self.dataset_name_to_cache_dir["librispeech_en_clean"],
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "ami_validation": load_dataset("edinburghcstr/ami",
                                           name="ihm",
                                           split="validation",
                                           cache_dir=self.cache_dir_ami,
                                           streaming=self.streaming),
            "ami": load_dataset("edinburghcstr/ami",
                                 name="ihm",
                                 split="test",
                                 cache_dir=self.cache_dir_ami,
                                 streaming=self.streaming),
            "tedlium": load_dataset(path="esb/diagnostic-dataset",
                                    name="tedlium",
                                    split="clean",
                                    cache_dir=self.dataset_name_to_cache_dir["tedlium"],
                                    streaming=self.streaming,
                                    use_auth_token=True).rename_column("norm_transcript", "text"),
            "librispeech_fr": load_dataset(path="facebook/multilingual_librispeech",
                                           name="french",
                                           split="test",
                                           cache_dir=self.dataset_name_to_cache_dir["librispeech_fr"],
                                           streaming=self.streaming,
                                           use_auth_token=True)
        }
        
        self.str2dataset = dicttoolz.keyfilter(lambda k: k in self.subset, self.str2dataset)
        
        # Remove unnecessary columns from the datasets:
        if "librispeech_en_clean" in self.str2dataset:
            self.str2dataset["librispeech_en_clean"] = remove_unnecessary_cols_for_librispeech(self.str2dataset["librispeech_en_clean"])
        if "ami" in self.str2dataset:
            self.str2dataset["ami"] = remove_unnecessary_cols_for_ami(self.str2dataset["ami"])
        
        return

    
    def _load_cache_dir_from_env_var(self) -> None:
        self.cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
        if self.cache_dir_librispeech is None:
            print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_librispeech}`.")
        
        self.cache_dir_esb = os.environ.get("CACHE_DIR_ESB_DIAGNOSTIC", None)
        if self.cache_dir_esb is None:
            print("WARNING: `CACHE_DIR_ESB_DIAGNOSTIC` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_esb}`.")
        
        self.cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
        if self.cache_dir_ami is None:
            print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_ami}`.")
        
        self.cache_dir_mls = os.environ.get("CACHE_DIR_MLS", None)
        if self.cache_dir_mls is None:
            print("WARNING: `CACHE_DIR_MLS` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_mls}`.")
        
        return
