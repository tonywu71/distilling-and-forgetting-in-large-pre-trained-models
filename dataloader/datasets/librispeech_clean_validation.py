from typing import Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup


class LibriSpeechCleanValidation(BaseDatasetGroup):
    """
    Util DatasetGroup to eval the vanilla Whisper model on the LibriSpeech clean validation set.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.available_datasets = [
            "librispeech_clean_validation"
        ]
        
        self.is_multilingual = False
        self.language = "english"
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_clean_validation": load_dataset(path="librispeech_asr",
                                                         name="clean",
                                                         split="validation",
                                                         streaming=self.streaming,
                                                         use_auth_token=True),
        }
