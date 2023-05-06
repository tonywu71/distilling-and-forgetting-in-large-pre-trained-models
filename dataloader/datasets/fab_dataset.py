from typing import Callable, Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from utils.constants import DEFAULT_NUM_PROC


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
            "librispeech_clean": load_dataset(path="librispeech_asr",
                                              name="clean",
                                              split="test",
                                              streaming=self.streaming,
                                              use_auth_token=True),
            "librispeech_other": load_dataset(path="librispeech_asr",
                                              name="other",
                                              split="test",
                                              streaming=self.streaming,
                                              use_auth_token=True),
            "tedlium": load_dataset(path="esb/diagnostic-dataset",
                                    name="tedlium",
                                    split="clean",
                                    streaming=self.streaming,
                                    use_auth_token=True),
            "librispeech_fr": load_dataset(path="facebook/multilingual_librispeech",
                                           name="french",
                                           split="test",
                                           streaming=self.streaming,
                                           use_auth_token=True)
        }
    
    
    def preprocess_datasets(self,
                            normalize_fct: Optional[Callable]=None) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        
        # Loop over all the datasets in the ESB benchmark:
        for dataset_name, dataset in self.str2dataset.items():
            # Normalize references (especially important for Whisper):
            if normalize_fct:
                dataset = dataset.map(normalize_fct, num_proc=DEFAULT_NUM_PROC)  # type: ignore
            # Update dataset:
            self.str2dataset[dataset_name] = dataset

        self.preprocessed = True
        return
