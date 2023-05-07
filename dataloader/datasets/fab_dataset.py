from typing import Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup


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
            "librispeech_fr_clean",
        ]
        
        self.is_multilingual = True
        self.ds_name_to_lang = {
            "librispeech_en_clean": "en",
            "librispeech_en_other": "en",
            "tedlium": "en",
            "librispeech_fr_clean": "fr"
        }
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_en_clean": load_dataset(path="librispeech_asr",
                                                 name="clean",
                                                 split="test",
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "librispeech_en_other": load_dataset(path="librispeech_asr",
                                                 name="other",
                                                 split="test",
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "tedlium": load_dataset(path="esb/diagnostic-dataset",
                                    name="tedlium",
                                    split="clean",
                                    streaming=self.streaming,
                                    use_auth_token=True
                                    ).rename_column("norm_transcript", "text"),
            "librispeech_fr": load_dataset(path="facebook/multilingual_librispeech",
                                           name="french",
                                           split="test",
                                           streaming=self.streaming,
                                           use_auth_token=True)
        }
