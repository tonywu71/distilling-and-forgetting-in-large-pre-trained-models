import os

from typing import Optional, List
from datasets import load_dataset, concatenate_datasets

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup


class MLSDataset(BaseDatasetGroup):
    """
    Class that regroups the Multilingual LibriSpeech (MLS) datasets.
    See for more details:
    - https://arxiv.org/abs/2012.03411
    - https://huggingface.co/datasets/facebook/multilingual_librispeech
    """
    
    def __init__(self,
                 streaming: bool=False,
                 load_diagnostic: bool=False,
                 subset: Optional[List[str]]=None) -> None:    
        super().__init__(streaming=streaming, subset=subset)
        self.load_diagnostic = load_diagnostic
        
        assert self.load_diagnostic and not self.streaming, "Streaming is not supported for the MLS-diagnostic set."
        
        # Set the abstract class attributes:
        self.dataset_path = "facebook/multilingual_librispeech"
        self.available_datasets = [
            "dutch",
            "english",
            "french",
            "german",
            "italian",
            "polish",
            "portuguese",
            "spanish"
        ]
        self.is_multilingual = True
        self.ds_name_to_lang = {
            "dutch": "dutch",
            "english": "english-basic_normalizer",  # use the multilingual normalizer for fair comparison
            "french": "french",
            "german": "german",
            "italian": "italian",
            "polish": "polish",
            "portuguese": "portuguese",
            "spanish": "spanish"
        }
        
        self._load_cache_dir_from_env_var()
        
        self.dataset_name_to_cache_dir = {
            "dutch": self.cache_dir_non_english_librispeech,
            "english": self.cache_dir_en_librispeech,
            "french": self.cache_dir_non_english_librispeech,
            "german": self.cache_dir_non_english_librispeech,
            "italian": self.cache_dir_non_english_librispeech,
            "polish": self.cache_dir_non_english_librispeech,
            "portuguese": self.cache_dir_non_english_librispeech,
            "spanish": self.cache_dir_non_english_librispeech
        }
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        if not self.load_diagnostic:  # If `load_diagnostic` default MLS dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    if dataset_name == "english":
                        # Load the 2 test splits of LibriSpeech from the original HF dataset as
                        # we want a fair comparison with the other datasets.
                        # Important note: `streaming` is set to `False` here as we want to take advantage
                        # of the fact that the full LibriSpeech dataset has already been cached.
                        librispeech_en_clean = load_dataset(path="librispeech_asr",
                                                            name="clean",
                                                            split="test",
                                                            cache_dir=self.dataset_name_to_cache_dir["english"],
                                                            streaming=False,
                                                            use_auth_token=True)
                        librispeech_en_other = load_dataset(path="librispeech_asr",
                                                            name="other",
                                                            split="test",
                                                            cache_dir=self.dataset_name_to_cache_dir["english"],
                                                            streaming=False,
                                                            use_auth_token=True)
                        self.str2dataset["english"] = concatenate_datasets([librispeech_en_clean, librispeech_en_other])  # type: ignore
                    
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                    name=dataset_name,
                                                                    split="test",
                                                                    cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                    streaming=self.streaming,
                                                                    use_auth_token=True)

        else:  # If load diagnostic dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:  # type: ignore
                    if dataset_name == "english":
                        librispeech_en_clean = load_dataset(path="librispeech_asr",
                                                            name="clean",
                                                            split="test[:180]",  # 90min
                                                            cache_dir=self.dataset_name_to_cache_dir["english"],
                                                            streaming=False,
                                                            use_auth_token=True)
                        librispeech_en_other = load_dataset(path="librispeech_asr",
                                                            name="other",
                                                            split="test[:180]",  # 90min
                                                            cache_dir=self.dataset_name_to_cache_dir["english"],
                                                            streaming=False,
                                                            use_auth_token=True)
                        self.str2dataset["english"] = concatenate_datasets(
                            [librispeech_en_clean, librispeech_en_other])  # 2*90 = 180min  # type: ignore
                        
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                    name=dataset_name,
                                                                    split="test[:120]",  # 60min
                                                                    cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                    streaming=self.streaming,
                                                                    use_auth_token=True)


    def _load_cache_dir_from_env_var(self) -> None:
        self.cache_dir_en_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
        if self.cache_dir_en_librispeech is None:
            print("WARNING: `CACHE_DIR_EN_LIBRISPEECH` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_en_librispeech}`.")
        
        self.cache_dir_non_english_librispeech = os.environ.get("CACHE_DIR_MLS", None)
        
        if self.cache_dir_non_english_librispeech is None:
            print("WARNING: `CACHE_DIR_MLS` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_non_english_librispeech}`.")

        return
