import os

from typing import Optional, List
from datasets import load_dataset

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.dataset_for_training.dataset_loader_librispeech import remove_unnecessary_cols_for_librispeech
from dataloader.dataset_for_training.dataset_loader_ami import remove_unnecessary_cols_for_ami


class ESBDiagnosticCustomDataset(BaseDatasetGroup):    
    """
    Custom version of the ESB-diagnostic dataset.
    
    The LibriSpeech and the AMI datasets from esb_diagnostic are replaced by the original
    LibriSpeech (clean and other) and AMI test splits.
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        super().__init__(streaming=streaming, subset=subset)
        
        # Set the abstract class attributes:
        self.dataset_path = "esb/diagnostic-dataset"
        self.available_datasets = [
            "librispeech",
            "common_voice",
            "voxpopuli",
            "tedlium",
            "gigaspeech",
            "spgispeech",
            "earnings22",
            "ami"
        ]
        self.is_multilingual = False
        self.language = "english"
        
        self._load_cache_dir_from_env_var()
        
        self.dataset_name_to_cache_dir = {
            "librispeech": self.cache_dir_librispeech,
            "common_voice": self.cache_dir_esb,
            "voxpopuli": self.cache_dir_esb,
            "tedlium": self.cache_dir_esb,
            "gigaspeech": self.cache_dir_esb,
            "spgispeech": self.cache_dir_esb,
            "earnings22": self.cache_dir_esb,
            "ami": self.cache_dir_ami,
        }
        
        self.post_init()
    
    
    def _prepare_str2dataset(self) -> None:
        for dataset_name in self.available_datasets:
            if dataset_name in self.subset:  # type: ignore
                if dataset_name == "librispeech":
                    self.str2dataset["librispeech_clean"] = load_dataset(path="librispeech_asr",
                                                                         name="clean",
                                                                         split="test",
                                                                         cache_dir=self.dataset_name_to_cache_dir["librispeech"],
                                                                         streaming=self.streaming,
                                                                         use_auth_token=True)
                    self.str2dataset["librispeech_other"] = load_dataset(path="librispeech_asr",
                                                                         name="other",
                                                                         split="test",
                                                                         cache_dir=self.dataset_name_to_cache_dir["librispeech"],
                                                                         streaming=self.streaming,
                                                                         use_auth_token=True)
                    self.str2dataset["librispeech_clean"] = remove_unnecessary_cols_for_librispeech(self.str2dataset["librispeech_clean"])
                    self.str2dataset["librispeech_other"] = remove_unnecessary_cols_for_librispeech(self.str2dataset["librispeech_other"])
                
                elif dataset_name == "ami":
                    self.str2dataset["ami"] = load_dataset("edinburghcstr/ami",
                                                           name="ihm",
                                                           split="test",
                                                           cache_dir=self.cache_dir_ami)
                    self.str2dataset["ami"] = remove_unnecessary_cols_for_ami(self.str2dataset["ami"])
                
                else:
                    self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                    name=dataset_name,
                                                                    split="clean",
                                                                    cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                    streaming=self.streaming,
                                                                    use_auth_token=True
                                                                    ).rename_column("norm_transcript", "text")

    
    def _load_cache_dir_from_env_var(self) -> None:
        self.cache_dir_esb = os.environ.get("CACHE_DIR_ESB_DIAGNOSTIC", None)
        if self.cache_dir_esb is None:
            print("WARNING: `CACHE_DIR_ESB_DIAGNOSTIC` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_esb}`.")
        
        self.cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
        if self.cache_dir_librispeech is None:
            print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_librispeech}`.")
        
        self.cache_dir_ami = os.environ.get("CACHE_DIR_AMI", None)
        if self.cache_dir_ami is None:
            print("WARNING: `CACHE_DIR_AMI` environment variable not set. Using default cache directory.")
        else:
            print(f"Using cache directory: `{self.cache_dir_ami}`.")

        return
