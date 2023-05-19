from typing import Optional, List
from datasets import load_dataset, concatenate_datasets

from dataloader.datasets.base_dataset_group import BaseDatasetGroup


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
        
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
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
                                                        streaming=False,
                                                        use_auth_token=True)
                    librispeech_en_other = load_dataset(path="librispeech_asr",
                                                        name="other",
                                                        split="test",
                                                        streaming=False,
                                                        use_auth_token=True)
                    self.str2dataset["english"] = concatenate_datasets([librispeech_en_clean, librispeech_en_other])  # type: ignore
                    
                else:
                    self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                  name=dataset_name,
                                                                  split="test",
                                                                  streaming=self.streaming,
                                                                  use_auth_token=True)
