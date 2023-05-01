from functools import partial
import string
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from transformers import WhisperFeatureExtractor, WhisperTokenizer
from datasets import Audio, Dataset, DatasetDict
from normalization.whisper_normalization import get_whisper_normalizer

from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_LABEL_TOKENIZED_COL


DEFAULT_NUM_PROC = 8  # see https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#hardware


# Audio augmentation object to map over the dataset:
AUGMENT_WAVEFORM = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3)
])


def extract_audio(dataset: Dataset, sampling_rate: int, audio_column: str="audio") -> Dataset:
    """Extract audio from dataset."""
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    return dataset


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


def normalize_sentence(sentence: str) -> str:
    """DEPRECATED. Normalize a sentence for transcription."""
    transcription = sentence
    
    if transcription.startswith('"') and transcription.endswith('"'):
        # Remove trailing quotation marks as they do not affect the transcription:
        transcription = transcription[1:-1]
    
    if transcription[-1] not in [".", "?", "!"]:
        # Append a full-stop to sentences that do not end in punctuation:
        transcription = transcription + "."
    
    transcription = transcription[:-1].translate(str.maketrans('', '', string.punctuation)) + transcription[-1]
    
    return transcription


def prepare_dataset_fct(dataset: DatasetDict,
                        tokenizer: WhisperTokenizer,
                        feature_extractor: WhisperFeatureExtractor) -> DatasetDict:
    """
    Utility to create features for a dataset.
    """    
    audio = dataset["audio"]
    
    # Extract features from audio (including log-Mel input features):
    # Note: the sampling rate arg is redundant but required to dismiss warnings.
    dataset["input_features"] = feature_extractor(
        audio["array"], sampling_rate=feature_extractor.sampling_rate).input_features[0]  # drop batch dimension
    
    # Encode from target text to label ids:
    dataset[DEFAULT_LABEL_TOKENIZED_COL] = tokenizer(dataset[DEFAULT_LABEL_STR_COL]).input_ids  # type: ignore
    
    return dataset


def filter_empty_strings(sentence) -> bool:
    """DEPRECATED. Filter nulls and short transcripts from dataset."""
    if len(sentence) < 2:
        return False
    else:
        return True


def preprocess_dataset(dataset_dict: DatasetDict,
                       tokenizer: WhisperTokenizer,
                       feature_extractor: WhisperFeatureExtractor,
                       augment: bool=False) -> DatasetDict:
    """
    Preprocess the dataset:
    - Extract audio from the dataset
    - Augment the dataset (optional)
    - Normalize the labels
    - Filter empty and short sentences
    - Prepare the dataset (extract features and tokenize labels)
    """
    
    for split in dataset_dict:
        dataset_dict[split] = extract_audio(dataset_dict[split],
                                            sampling_rate=feature_extractor.sampling_rate,
                                            audio_column="audio")
        
        if augment and split == "train":  # only augment the training set
            augment_dataset = partial(augment_dataset_fct, sample_rate=feature_extractor.sampling_rate)
            dataset_dict[split] = dataset_dict[split].map(augment_dataset, num_proc=DEFAULT_NUM_PROC)
        
        # Apply Whisper's normalization to the labels:
        # This operation must be done before the dataset is prepared as the
        # tokenizer should be applied to the normalized text.
        whisper_norm = get_whisper_normalizer(tokenizer)
        
        def normalize_fct(batch):
            batch[DEFAULT_LABEL_STR_COL] = whisper_norm(batch[DEFAULT_LABEL_STR_COL])
            return batch

        dataset_dict[split] = dataset_dict[split].map(normalize_fct, num_proc=DEFAULT_NUM_PROC)
        
        prepare_dataset = partial(prepare_dataset_fct,
                                  tokenizer=tokenizer,
                                  feature_extractor=feature_extractor)
        
        dataset_dict[split] = dataset_dict[split].map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)

    return dataset_dict
