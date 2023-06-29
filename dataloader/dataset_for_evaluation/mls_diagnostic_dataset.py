from typing import Optional, List
from datasets import load_dataset, concatenate_datasets

from dataloader.dataset_for_evaluation.mls_dataset import MLSDataset



class MLSDiagnosticDataset(MLSDataset):
    """
    Class that regroups the Multilingual LibriSpeech (MLS) datasets.
    See for more details:
    - https://arxiv.org/abs/2012.03411
    - https://huggingface.co/datasets/facebook/multilingual_librispeech
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:    
        assert not self.streaming, "Streaming is not supported for MLSDiagnosticDataset."
        super().__init__(streaming=streaming, subset=subset)
    
    
    def _prepare_str2dataset(self) -> None:
        """
        Override `_prepare_str2dataset` from MLSDataset.
        """
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
