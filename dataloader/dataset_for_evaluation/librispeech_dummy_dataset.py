from typing import Optional, List
import os
from datasets import load_dataset
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup


class LibriSpeechDummyDataset(BaseDatasetGroup):
    """
    Debug DatasetGroup with the lightweight validation set from `hf-internal-testing/librispeech_asr_dummy`.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        super().__init__(streaming=streaming, subset=subset)
        
        # Set the abstract class attributes:
        self.available_datasets = [
            "librispeech_dummy"
        ]
        self.is_multilingual = False
        self.language = "english"
        
        self.post_init()
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_dummy": load_dataset("hf-internal-testing/librispeech_asr_dummy",
                                              name="clean",
                                              split="validation",
                                              cache_dir=self.cache_dir_librispeech)
        }


    def _load_cache_dir_from_env_var(self) -> None:
        self.cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
        if self.cache_dir_librispeech is None:
            print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_librispeech}`.")
