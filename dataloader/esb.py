from typing import Callable, Optional, List
from datasets import load_dataset


class ESB_Datasets:
    """
    Class that encompasses the End-to-end Speech Benchmark (ESB)
    See for more details:
    - https://arxiv.org/abs/2210.13352 
    - https://huggingface.co/datasets/esb/datasets
    - https://huggingface.co/datasets/esb/diagnostic-dataset
    """
    
    def __init__(self,
                 streaming: bool=False,
                 load_diagnostic: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
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
        
        self.streaming = streaming
        self.load_diagnostic = load_diagnostic
        self.subset = subset
        
        dataset_path = "esb/datasets" if not load_diagnostic else "esb/diagnostic-dataset"
        
        if not self.subset:
            self.subset = self.available_datasets  # if no subset is specified, use all datasets
        else:
            assert all([k in self.str2dataset for k in self.available_datasets]), f"`subset` must be a subset of {list(self.str2dataset.keys())}."
        
        self.str2dataset = {}
        
        if not self.load_diagnostic:  # If load default ESB dataset...
            for dataset_name in self.available_datasets:
                if dataset_name in self.subset:
                    if dataset_name == "librispeech":
                        self.str2dataset["librispeech_clean"] = load_dataset(path=dataset_path,
                                                                                name=dataset_name,
                                                                                split="test.clean",
                                                                                streaming=self.streaming,
                                                                                use_auth_token=True)
                        self.str2dataset["librispeech_clean"] = load_dataset(path=dataset_path,
                                                                                name=dataset_name,
                                                                                split="test.other",
                                                                                streaming=self.streaming,
                                                                                use_auth_token=True)
                    else:
                        self.str2dataset[dataset_name] = load_dataset(path=dataset_path,
                                                                        name=dataset_name,
                                                                        split="test",
                                                                        streaming=self.streaming,
                                                                        use_auth_token=True)
        
        else:  # If load diagnostic dataset...
            for dataset_name in self.available_datasets:
                self.str2dataset[dataset_name] = load_dataset(path=dataset_path,
                                                              name=dataset_name,
                                                              split="clean",
                                                              streaming=self.streaming,
                                                              use_auth_token=True)
            
        self.preprocessed = False
    
        
    def preprocess_datasets(self,
                            normalize_fct: Optional[Callable]=None) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        
        # Loop over all the datasets in the ESB benchmark:
        for dataset_name, dataset in self.str2dataset.items():
            if self.load_diagnostic:
                dataset = dataset.rename_column("norm_transcript", "text")
            
            # Normalize references (especially important for Whisper):
            if normalize_fct:
                dataset = dataset.map(normalize_fct)
            
            # Update dataset:
            self.str2dataset[dataset_name] = dataset

        self.preprocessed = True
        
        return
    
    
    def __getitem__(self, dataset_name):
        return self.str2dataset[dataset_name]
    
    
    def keys(self):
        return self.str2dataset.keys()
    
    
    def items(self):
        return self.str2dataset.items()
