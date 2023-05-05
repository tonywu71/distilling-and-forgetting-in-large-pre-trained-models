from typing import Optional, List, Callable, Dict
from abc import ABC, abstractmethod


class BaseDatasetGroup(ABC):
    """
    Base class used to handle a group of datasets.
    """
    
    # Must be set in the child class before calling `super().__init__()`:
    available_datasets: List[str] = []
    is_multilingual: bool = False
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
        
        self.preprocessed = False
        if self.is_multilingual:
            assert hasattr(self, "ds_name_to_lang"), "If `is_multilingual` is True, `ds_name_to_lang` must be set in the child class before calling `super().__init__()`."
            assert set(self.available_datasets) == set(self.ds_name_to_lang.keys()), "`ds_name_to_lang` must have the same keys as `self.available_datasets`."
    
    
    @abstractmethod
    def _prepare_str2dataset(self) -> None:
        """
        Must be called after `super().__init__()`.
        """
        pass
    
    
    @abstractmethod
    def preprocess_datasets(self, normalize_fct: Optional[Callable]=None, **kwargs) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        pass
    
    
    def __getitem__(self, dataset_name: str):
        return self.str2dataset[dataset_name]
    
    
    def keys(self):
        return self.str2dataset.keys()
    
    
    def items(self):
        return self.str2dataset.items()
