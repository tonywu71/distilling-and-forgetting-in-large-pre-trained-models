from typing import Optional, List
import os
from toolz import dicttoolz
from datasets import load_dataset
from dataloader.dataset_for_evaluation.fab_dataset import FABDataset
from dataloader.dataset_for_training.dataset_loader_librispeech import remove_unnecessary_cols_for_librispeech
from dataloader.dataset_for_training.dataset_loader_ami import remove_unnecessary_cols_for_ami



class FABDiagnosticDataset(FABDataset):
    """
    Diagnostic version of the FAB dataset. Should be used for debug only.
    """
    
    def _prepare_str2dataset(self) -> None:
        self.str2dataset = {
            "librispeech_en_clean": load_dataset(path="librispeech_asr",
                                                 name="clean",
                                                 split="test[:10%]",
                                                 cache_dir=self.dataset_name_to_cache_dir["librispeech_en_clean"],
                                                 streaming=self.streaming,
                                                 use_auth_token=True),
            "ami": load_dataset("edinburghcstr/ami",
                                 name="ihm",
                                 split="test[:10%]",
                                 cache_dir=self.cache_dir_ami,
                                 streaming=self.streaming),
            "tedlium": load_dataset(path="esb/diagnostic-dataset",
                                    name="tedlium",
                                    split="clean",
                                    cache_dir=self.dataset_name_to_cache_dir["tedlium"],
                                    streaming=self.streaming,
                                    use_auth_token=True).rename_column("norm_transcript", "text"),
            "librispeech_fr": load_dataset(path="facebook/multilingual_librispeech",
                                           name="french",
                                           split="test[:20%]",
                                           cache_dir=self.dataset_name_to_cache_dir["librispeech_fr"],
                                           streaming=self.streaming,
                                           use_auth_token=True),
            "librispeech_pt": load_dataset(path="facebook/multilingual_librispeech",
                                           name="portuguese",
                                           split="test[:20%]",
                                           cache_dir=self.dataset_name_to_cache_dir["librispeech_pt"],
                                           streaming=self.streaming,
                                           use_auth_token=True)
        }
        
        self.str2dataset = dicttoolz.keyfilter(lambda k: k in self.subset, self.str2dataset)
        
        # Remove unnecessary columns from the datasets:
        if "librispeech_en_clean" in self.str2dataset:
            self.str2dataset["librispeech_en_clean"] = remove_unnecessary_cols_for_librispeech(self.str2dataset["librispeech_en_clean"])
        if "ami" in self.str2dataset:
            self.str2dataset["ami"] = remove_unnecessary_cols_for_ami(self.str2dataset["ami"])
        
        return
