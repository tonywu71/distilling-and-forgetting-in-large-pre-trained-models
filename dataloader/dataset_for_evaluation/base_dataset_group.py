from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

from dataloader.filtering import filter_audio_length


class BaseDatasetGroup(ABC):
    """
    Base class used to handle a group of datasets.
    """
    
    def __init__(self,
                 streaming: bool = False,
                 subset: Optional[List[str]] = None) -> None:
        self.streaming = streaming
        self.subset = subset
        self.str2dataset = {}
        
        # Abstract attributes:
        self.available_datasets: Optional[List[str]] = None
        self.is_multilingual: Optional[bool] = None
        self.language: Optional[str] = None  # language should be left to None if `is_multilingual` is True
        self.ds_name_to_lang: Optional[Dict[str, str]] = None  # ds_name_to_lang should be left to None if `is_multilingual` is False
        
    
    def post_init(self) -> None:
        """
        Post-initialization steps.
        """
        
        # Sanity checks:
        self.assert_abstract_attributes_are_set()
        self.assert_langauge_attributes_are_correct()
        
        # Set subset to all datasets if not specified:
        if self.subset:
            assert all([k in self.available_datasets for k in self.subset]), f"`subset` must be a subset of {self.available_datasets}."
        else:
            self.subset = self.available_datasets  # if no subset is specified, use all datasets
        
        # Fill `self.str2dataset` depending on the datasets to load:
        self._load_cache_dir_from_env_var()
        self._prepare_str2dataset()
    
    
    def assert_abstract_attributes_are_set(self):
        """
        Assert that the abstract attributes are set.
        """
        for attr in ["available_datasets", "is_multilingual"]:
            assert getattr(self, attr) is not None, f"`{attr}` must be set in the child class before calling `super().__init__()`."
    
    
    def assert_langauge_attributes_are_correct(self):
        if self.is_multilingual:
            assert getattr(self, "ds_name_to_lang") is not None, "If `is_multilingual` is True, `ds_name_to_lang` must be set in the child class before calling `super().__init__()`."
            assert set(self.available_datasets) == set(self.ds_name_to_lang.keys()), "`ds_name_to_lang` must have the same keys as `self.available_datasets`."
        else:
            assert self.language is not None, "If `is_multilingual` is False, `language` must be set in the child class before calling `super().__init__()`."
    
    
    @abstractmethod
    def _load_cache_dir_from_env_var(self) -> None:
        """
        Load the cache directories from the environment variables.
        """
        pass
        
    
    @abstractmethod
    def _prepare_str2dataset(self) -> None:
        """
        Load the different datasets. Will be automatically called in `super().__init__()`.
        """
        pass
    
    
    def filter_audio_length(self, verbose: bool = True) -> None:
        """
        Filter the audio files that are too short or too long.
        Not necessary if the dataset is already suitable for the model at hand.
        """
        tbar = tqdm(self.str2dataset, disable=not(verbose))
        
        for dataset_name in tbar:  # Loop over all the datasets...
            tbar.set_description(f"Processing `{dataset_name}`...")
            self.str2dataset[dataset_name] = filter_audio_length(self.str2dataset[dataset_name])
        
        return
    
    
    def __getitem__(self, dataset_name: str):
        return self.str2dataset[dataset_name]
    
    
    def keys(self):
        return self.str2dataset.keys()
    
    
    def items(self):
        return self.str2dataset.items()
