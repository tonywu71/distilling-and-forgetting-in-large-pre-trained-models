from typing import Optional, List, Dict
from abc import ABC, abstractmethod

from dataloader.filtering import filter_audio_length, filter_labels
from normalization.whisper_normalization import get_whisper_normalizer
from utils.constants import DEFAULT_NUM_PROC


class BaseDatasetGroup(ABC):
    """
    Base class used to handle a group of datasets.
    """
    
    # Must be set in the child class before calling `super().__init__()`:
    available_datasets: List[str] = []
    is_multilingual: bool = False
    language: Optional[str] = None
    ds_name_to_lang: Dict[str, str] = {}
    
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        self.streaming = streaming
        self.subset = subset
        
        self.str2dataset = {}
        
        if not self.subset:
            self.subset = self.available_datasets  # if no subset is specified, use all datasets
        else:
            assert all([k in self.available_datasets for k in self.subset]), f"`subset` must be a subset of {self.available_datasets}."
        
        # Fill `self.str2dataset` depending on the datasets to load:
        self._prepare_str2dataset()
        
        self.is_preprocessed = False
        if self.is_multilingual:
            assert hasattr(self, "ds_name_to_lang"), "If `is_multilingual` is True, `ds_name_to_lang` must be set in the child class before calling `super().__init__()`."
            assert set(self.available_datasets) == set(self.ds_name_to_lang.keys()), "`ds_name_to_lang` must have the same keys as `self.available_datasets`."
        else:
            assert self.language is not None, "If `is_multilingual` is False, `language` must be set in the child class before calling `super().__init__()`."
    
    
    @abstractmethod
    def _prepare_str2dataset(self) -> None:
        """
        Must be called after `super().__init__()`.
        """
        pass
    
    
    def preprocess_datasets(self, normalize: bool=True) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.is_preprocessed, "Datasets have already been preprocessed."
        
        if not self.is_multilingual:
            # Load normalizer before the loop to avoid loading it multiple times:
            whisper_norm = get_whisper_normalizer(language=self.language)
            
            def normalize_fct(batch):
                batch["text"] = whisper_norm(batch["text"])
                return batch
        
            # Loop over all the datasets:
            for dataset_name in self.str2dataset.items():
                # Filter audio length and labels:
                dataset = filter_audio_length(dataset)
                dataset = filter_labels(dataset)
                
                # Normalize the labels:
                if normalize:
                    if not self.streaming:
                        dataset = dataset.map(normalize_fct, num_proc=DEFAULT_NUM_PROC)  # type: ignore
                    else:
                        dataset = dataset.map(normalize_fct)  # type: ignore
                
                # Update dataset:
                self.str2dataset[dataset_name] = dataset
        
        else:  # If multilingual...
            # Loop over all the datasets:
            for dataset_name, dataset in self.str2dataset.items():
                # Filter audio length and labels:
                dataset = filter_audio_length(dataset)
                dataset = filter_labels(dataset)
                
                # Load normalizer depending on the language:
                whisper_norm = get_whisper_normalizer(language=self.ds_name_to_lang[dataset_name])
                def normalize_fct(batch):
                    batch["text"] = whisper_norm(batch["text"])
                    return batch
                
                # Normalize the labels:
                if not self.streaming:
                    dataset = dataset.map(normalize_fct, num_proc=DEFAULT_NUM_PROC)  # type: ignore
                else:
                    dataset = dataset.map(normalize_fct)  # type: ignore
                
                # Update dataset:
                self.str2dataset[dataset_name] = dataset

        self.is_preprocessed = True
        return
    
    
    def __getitem__(self, dataset_name: str):
        return self.str2dataset[dataset_name]
    
    
    def keys(self):
        return self.str2dataset.keys()
    
    
    def items(self):
        return self.str2dataset.items()
