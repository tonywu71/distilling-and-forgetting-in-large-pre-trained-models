from typing import Optional, List
from datasets import load_dataset, concatenate_datasets

from dataloader.dataset_for_evaluation.mls_dataset import MLSDataset



class MLSDiagnosticDatasetCustom(MLSDataset):
    """
    Custom and lightweight version of the MLS dataset (MLSDC).
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:    
        assert not streaming, "Streaming is not supported for MLSDiagnosticDatasetCustom."
        super().__init__(streaming=False, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        """
        Override `_prepare_str2dataset` from MLSDataset.
        """
        for dataset_name in self.available_datasets:
            if dataset_name in self.subset:  # type: ignore
                if dataset_name == "english":
                    librispeech_en_clean = load_dataset(path="librispeech_asr",
                                                        name="clean",
                                                        split="test[:10%]",
                                                        cache_dir=self.dataset_name_to_cache_dir["english"],
                                                        streaming=False,
                                                        use_auth_token=True)
                    librispeech_en_other = load_dataset(path="librispeech_asr",
                                                        name="other",
                                                        split="test[:10%]",
                                                        cache_dir=self.dataset_name_to_cache_dir["english"],
                                                        streaming=False,
                                                        use_auth_token=True)
                    self.str2dataset["english"] = concatenate_datasets(
                        [librispeech_en_clean, librispeech_en_other])

                else:
                    self.str2dataset[dataset_name] = load_dataset(path=self.dataset_path,
                                                                name=dataset_name,
                                                                split="test[:20%]",
                                                                cache_dir=self.dataset_name_to_cache_dir[dataset_name],
                                                                streaming=self.streaming,
                                                                use_auth_token=True)
