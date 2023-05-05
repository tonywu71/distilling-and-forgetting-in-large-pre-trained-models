from functools import partial
from typing import Any, Dict
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from transformers import WhisperFeatureExtractor, WhisperTokenizer
from datasets import Audio, DatasetDict
from normalization.whisper_normalization import get_whisper_normalizer

from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_LABEL_TOKENIZED_COL, DEFAULT_NUM_PROC


# Both in seconds:
MIN_INPUT_LENGTH = 0.0
MAX_INPUT_LENGTH = 30.0


# Audio augmentation object to map over the dataset:
AUGMENT_WAVEFORM = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3)
])


def augment_dataset_fct(batch, sample_rate: int):
    """
    Perform data augmentation for audio.
    
    Notes:
        - `extract_audio` must be called before this function
        - should only be applied to the train set
    """
    audio = batch["audio"]["array"]
    augmented_audio = AUGMENT_WAVEFORM(samples=audio, sample_rate=sample_rate)
    batch["audio"]["array"] = augmented_audio
    return batch



def prepare_dataset_fct(batch: Dict[str, Any],
                        tokenizer: WhisperTokenizer,
                        feature_extractor: WhisperFeatureExtractor) -> Dict[str, Any]:
    """
    Utility to create features for a dataset.
    """    
    audio = batch["audio"]
    
    # Extract features from audio (including log-Mel input features):
    # Note: the sampling rate arg is redundant but required to dismiss warnings.
    batch["input_features"] = feature_extractor(audio["array"],
                                                sampling_rate=feature_extractor.sampling_rate).input_features[0]  # drop batch dimension
    
    # Encode from target text to label ids:
    batch[DEFAULT_LABEL_TOKENIZED_COL] = tokenizer(batch[DEFAULT_LABEL_STR_COL]).input_ids  # type: ignore
    
    return batch


def is_in_length_range(audio: Dict[str, Any], untokenized_text_label: list) -> bool:
    # Compute input length of audio sample in seconds:
    input_length = len(audio["array"]) / audio["sampling_rate"]  # type: ignore
    return MIN_INPUT_LENGTH < input_length < MAX_INPUT_LENGTH and 0 < len(untokenized_text_label)


def preprocess_dataset(dataset_dict: DatasetDict,
                       tokenizer: WhisperTokenizer,
                       feature_extractor: WhisperFeatureExtractor,
                       augment: bool=False) -> DatasetDict:
    """
    Preprocess the dataset:
    - Extract audio from the dataset
    - Augment the dataset (optional)
    - Normalize the labels
    - Prepare the dataset (extract features and tokenize labels)
    - Filter the dataset (remove ampty samples and samples that are too long)
    """
    
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        
        if augment and split == "train":  # only augment the training set
            print("Augmenting the training set...")
            augment_dataset = partial(augment_dataset_fct, sample_rate=feature_extractor.sampling_rate)
            dataset_dict[split] = dataset_dict[split].map(augment_dataset, num_proc=DEFAULT_NUM_PROC)
        
        # Apply Whisper's normalization to the labels:
        # This operation must be done before the dataset is prepared as the
        # tokenizer should be applied to the normalized text.
        whisper_norm = get_whisper_normalizer(tokenizer)
        
        def normalize_fct(batch):
            batch[DEFAULT_LABEL_STR_COL] = whisper_norm(batch[DEFAULT_LABEL_STR_COL])
            return batch
        
        print("Normalizing the labels...")
        dataset_dict[split] = dataset_dict[split].map(normalize_fct, num_proc=DEFAULT_NUM_PROC)
        
        prepare_dataset = partial(prepare_dataset_fct,
                                  tokenizer=tokenizer,
                                  feature_extractor=feature_extractor)
                
        print("Preparing the dataset...")
        dataset_dict[split] = dataset_dict[split].map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)
        
        print("Filtering the dataset...")
        dataset_dict[split] = dataset_dict[split].filter(
            is_in_length_range,
            input_columns=["audio", DEFAULT_LABEL_STR_COL],
            num_proc=DEFAULT_NUM_PROC
        )
    
    return dataset_dict
