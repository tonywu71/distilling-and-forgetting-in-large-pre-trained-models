import os

from typing import Optional, List
from datasets import load_dataset

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup


class FABDataset(BaseDatasetGroup):
    """
    Class that regroups a few datasets as part of the Forgetting Assessment Benchmark (FAB).
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.available_datasets = [
            "librispeech_en_clean",
            "librispeech_en_other",
            "tedlium",
            "librispeech_fr",
        ]
        
        self.is_multilingual = True
        self.ds_name_to_lang = {
            "librispeech_en_clean": "en",
            "librispeech_en_other": "en",
            "tedlium": "en",
            "librispeech_fr": "fr"
        }
        
        
        # Retrieve custom `cache_dir` filepath if set:
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
        
        self.cache_dir_mls = os.environ.get("CACHE_DIR_MLS", None)
        if self.cache_dir_mls is None:
            print("WARNING: `CACHE_DIR_MLS` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_mls}`.")
        
        
        self.dataset_name_to_cache_dir = {
            "librispeech_en_clean": self.cache_dir_librispeech,
            "librispeech_en_other": self.cache_dir_librispeech,
            "tedlium": self.cache_dir_esb,
            "librispeech_fr": self.cache_dir_mls,
        }
        
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_en_clean": load_dataset(path="librispeech_asr",
                                                 name="clean",
                                                 split="test",
                                                 cache_dir=self.dataset_name_to_cache_dir["librispeech_en_clean"],
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "librispeech_en_other": load_dataset(path="librispeech_asr",
                                                 name="other",
                                                 split="test",
                                                 cache_dir=self.dataset_name_to_cache_dir["librispeech_en_other"],
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "tedlium": load_dataset(path="esb/diagnostic-dataset",
                                    name="tedlium",
                                    split="clean",
                                    cache_dir=self.dataset_name_to_cache_dir["tedlium"],
                                    streaming=self.streaming,
                                    use_auth_token=True
                                    ).rename_column("norm_transcript", "text"),
            "librispeech_fr": load_dataset(path="facebook/multilingual_librispeech",
                                           name="french",
                                           split="test",
                                           cache_dir=self.dataset_name_to_cache_dir["librispeech_fr"],
                                           streaming=self.streaming,
                                           use_auth_token=True)
        }
