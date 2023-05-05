from typing import Callable, Optional, List
from datasets import load_dataset

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from utils.constants import DEFAULT_NUM_PROC


class MLSDataset(BaseDatasetGroup):
    """
    Class that regroups the Multilingual LibriSpeech (MLS) datasets.
    See for more details:
    - https://arxiv.org/abs/2012.03411
    - https://huggingface.co/datasets/facebook/multilingual_librispeech
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        self.dataset_path = "facebook/multilingual_librispeech"
        self.available_datasets = [
            "dutch",
            "french",
            "german",
            "italina",
            "polish",
            "portuguese",
            "spanish"
        ]
        
        self.is_multilingual = True
        self.ds_name_to_lang = {
            "dutch": "dutch",
            "french": "french",
            "german": "german",
            "italina": "italina",
            "polish": "polish",
            "portuguese": "portuguese",
            "spanish": "spanish"
        }
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        for dataset_name in self.available_datasets:
            if dataset_name in self.subset:  # type: ignore
                self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                              name=dataset_name,
                                                              split="test",
                                                              streaming=self.streaming,
                                                              use_auth_token=True)
    
        
    def preprocess_datasets(self, normalize_fct: Optional[Callable]=None) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        
        # Loop over all the datasets in the ESB benchmark:
        for dataset_name, dataset in self.str2dataset.items():
            # Normalize references (especially important for Whisper):
            if normalize_fct:
                dataset = dataset.map(normalize_fct, num_proc=DEFAULT_NUM_PROC)
            # Update dataset:
            self.str2dataset[dataset_name] = dataset

        self.preprocessed = True
        return
