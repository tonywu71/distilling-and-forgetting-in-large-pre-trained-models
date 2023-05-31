from typing import Optional, List

import os

from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup


class LibriSpeechCleanTestSet(BaseDatasetGroup):
    """
    Util DatasetGroup to eval the vanilla Whisper model on the LibriSpeech clean test set.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.available_datasets = [
            "librispeech_clean_test"
        ]
        
        self.is_multilingual = False
        self.language = "english"
        
        # Retrieve custom `cache_dir` filepath if set:
        self.cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
        if self.cache_dir_librispeech is None:
            print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_librispeech}`.")
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_clean_test": load_dataset(path="librispeech_asr",
                                                   name="clean",
                                                   split="test",
                                                   streaming=self.streaming,
                                                   cache_dir=self.cache_dir_librispeech)
        }
