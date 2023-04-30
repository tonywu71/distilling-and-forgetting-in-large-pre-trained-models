from typing import Callable, Optional, List
from datasets import load_dataset, Audio

from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_LABEL_TOKENIZED_COL


class ESB_Datasets:
    """
    Class that encompasses the End-to-end Speech Benchmark (ESB)
    See https://arxiv.org/abs/2210.13352 for more details.
    """
    
    def __init__(self, subset: Optional[List[str]]=None, no_auth_datasets_only: bool=False) -> None:
        self.subset = subset
        
        self.librispeech_clean = load_dataset("librispeech_asr", "all", split="test.clean", streaming=True)
        self.librispeech_other = load_dataset("librispeech_asr", "all", split="test.other", streaming=True)
        self.voxpopuli = load_dataset("facebook/voxpopuli", "en", split="test", streaming=True)
        self.tedlium = load_dataset("LIUM/tedlium", "release3", split="test", streaming=True)
        self.earnings22 = load_dataset("anton-l/earnings22_baseline_5_gram", split="test", streaming=True)
        self.ami = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
        
        if not no_auth_datasets_only:
            self.common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", revision="streaming", split="test", streaming=True, use_auth_token=True)
            self.gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", split="test", streaming=True, use_auth_token=True)
            self.spgispeech = load_dataset("kensho/spgispeech", "S", split="test", streaming=True, use_auth_token=True)
    
        self.str2dataset = {
            "LibriSpeech Clean": self.librispeech_clean,
            "LibriSpeech Other": self.librispeech_other,
            "VoxPopuli": self.voxpopuli,
            "TEDLIUM": self.tedlium,
            "Earnings-22": self.earnings22,
            "AMI": self.ami
        }
        
        if not no_auth_datasets_only:
            self.str2dataset.update({
                "Common Voice": self.common_voice,
                "GigaSpeech": self.gigaspeech,
                "SPGISpeech": self.spgispeech
        })
        
        if self.subset:
            assert all([k in self.str2dataset for k in self.subset]), f"`subset` must be a subset of {list(self.str2dataset.keys())}."
            self.str2dataset = {k: v for k, v in self.str2dataset.items() if k in self.subset}
        
        self.filter_sequences = [
            "ignore time segment in scoring",  # can be found in TEDLIUM for example
            ""
        ]
        
        self.preprocessed = False
    
        
    def preprocess_datasets(self,
                            sampling_rate: int,
                            normalize_fct: Optional[Callable]=None,
                            n_samples: Optional[int]=None) -> None:
        """
        Preprocess the datasets.
        """
        assert not self.preprocessed, "Datasets have already been preprocessed."
        
        # Loop over all the datasets in the ESB benchmark:
        for dataset_name, dataset in self.str2dataset.items():
            if n_samples:
                dataset = dataset.take(n_samples)  # type: ignore

            # Resample:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

            # Normalize references
            if normalize_fct:
                dataset = dataset.map(normalize_fct)

            # Remove any empty references:
            dataset = dataset.filter(self.is_target_text_in_range, input_columns=[DEFAULT_LABEL_STR_COL])
            
            # Update dataset
            self.str2dataset[dataset_name] = dataset

        self.preprocessed = True
        return
    
    
    def __getitem__(self, dataset_name):
        return self.str2dataset[dataset_name]
    
    
    def keys(self):
        return self.str2dataset.keys()
    
    
    def items(self):
        return self.str2dataset.items()
    
    
    @staticmethod
    def get_text(sample: dict) -> str:
        """
        Get the correct transcription column from the ESB datasets.
        """
        if "text" in sample:
            return sample["text"]
        elif "sentence" in sample:
            return sample["sentence"]
        elif "normalized_text" in sample:
            return sample["normalized_text"]
        elif "transcript" in sample:
            return sample["transcript"]
        elif DEFAULT_LABEL_TOKENIZED_COL in sample:
            return sample[DEFAULT_LABEL_TOKENIZED_COL]
        else:
            raise ValueError(f"Sample: {sample.keys()} has no transcript.")

    
    def is_target_text_in_range(self, ref: str) -> bool:
        ref = ref.strip()
        return ref not in self.filter_sequences
