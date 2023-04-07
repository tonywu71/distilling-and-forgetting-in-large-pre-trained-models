import string
from typing import Optional
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from transformers import WhisperFeatureExtractor, WhisperProcessor
from datasets import Audio, Dataset


# audio augmentation function to map over the dataset
AUGMENT_WAVEFORM = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3)
])


def normalize_sentence(sentence: str) -> str:
    """Normalize a sentence for transcription."""
    transcription = sentence
    
    if transcription.startswith('"') and transcription.endswith('"'):
        # Remove trailing quotation marks as they do not affect the transcription:
        transcription = transcription[1:-1]
    
    if transcription[-1] not in [".", "?", "!"]:
        # Append a full-stop to sentences that do not end in punctuation:
        transcription = transcription + "."
    
    transcription = transcription[:-1].translate(str.maketrans('', '', string.punctuation)) + transcription[-1]
    
    return transcription


def prepare_dataset(dataset: dict,
                    feature_extractor: WhisperFeatureExtractor,
                    tokenizer: WhisperProcessor):
    """utility to create features for a dataset"""
    
    audio = dataset["audio"]
    
    # Extract features from audio (including log-Mel input features)
    dataset["input_features"] = feature_extractor(audio["array"]).input_features[0]  # drop batch dimension
    
    # Normalize the transcription:
    sentences = normalize_sentence(dataset["sentence"])
    
    # Encode from target text to label ids:
    dataset["labels"] = tokenizer(sentences, max_length=225, truncation=True).input_ids
    
    return dataset


def filter_empty_strings(sentence) -> bool:
    """Filter nulls and short transcripts from dataset."""
    if len(sentence) < 2:
        return False
    else:
        return True


def extract_audio(dataset: Dataset, audio_column: str, sampling_rate: int):
    """Extract audio from dataset."""
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    return dataset


def augment_dataset(batch):
    """
    Perform data augmentation for audio.
    
    Notes:
        - `extract_audio` must be called before this function
        - should only be applied to the train set
    """
    audio = batch["audio"]["array"]
    # apply augmentation
    augmented_audio = AUGMENT_WAVEFORM(samples=audio, sample_rate=16000)

    batch["audio"]["array"] = augmented_audio

    return batch


def preprocess_dataset(dataset_dict,
                       feature_extractor: WhisperFeatureExtractor):
    """Preprocess the dataset."""
    
    for split in dataset_dict:
        dataset_dict[split] = extract_audio(dataset_dict[split],
                                        audio_column="audio",
                                        sampling_rate=feature_extractor.sampling_rate)
        
        if split == "train":  # only augment the training set
            dataset_dict[split] = dataset_dict[split].map(augment_dataset)
        
        dataset_dict[split] = dataset_dict[split].filter(filter_empty_strings, input_columns=["sentence"])\
                                         .map(prepare_dataset)

    return dataset_dict
