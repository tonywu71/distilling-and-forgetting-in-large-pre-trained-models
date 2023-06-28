from functools import partial
from typing import Any, Dict

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast
from datasets import Audio, DatasetDict
from dataloader.filtering import filter_audio_length, filter_labels

from dataloader.preprocessing_train.augmentation import augment_audio_fct
from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_LABEL_TOKENIZED_COL, DEFAULT_NUM_PROC


def lowercase_fct(example: Dict[str, str]) -> Dict[str, str]:
    return {DEFAULT_LABEL_STR_COL: example[DEFAULT_LABEL_STR_COL].lower()}


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


def preprocess_dataset(dataset_dict: DatasetDict,
                       tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                       feature_extractor: WhisperFeatureExtractor,
                       lowercase: bool = True,
                       augment: bool = False) -> DatasetDict:
    """
    Preprocess the dataset:
    - Extract audio from the dataset
    - Augment the dataset (optional)
    - Lowercase the text labels (optional)
    - Prepare the dataset (extract features and tokenize labels)
    - Filter the dataset (remove samples with audio with not suitable length and samples with an empty label).
    
    Important: Make sure that the label column has the the same name as `DEFAULT_LABEL_STR_COL`
               ("text" by default).
    
    Note: We deliberately chose to keep the audio features to be able to plot
          to log them in the wandb callback.
    """
    
    for split in dataset_dict:
        print(f"Preprocessing the {split} set...")
        
        print("Extracting audio from the dataset...")
        dataset_dict[split] = dataset_dict[split].cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        
        if augment and split == "train":  # only augment the training set
            print("Augmenting the training set...")
            augment_dataset = partial(augment_audio_fct, sample_rate=feature_extractor.sampling_rate)
            dataset_dict[split] = dataset_dict[split].map(augment_dataset, num_proc=DEFAULT_NUM_PROC)
        
        if lowercase:
            print("Lowercasing the dataset...")
            dataset_dict[split] = dataset_dict[split].map(lowercase_fct, num_proc=DEFAULT_NUM_PROC)
        
        print("Preparing the dataset...")
        prepare_dataset = partial(prepare_dataset_fct,
                                  tokenizer=tokenizer,
                                  feature_extractor=feature_extractor)
        dataset_dict[split] = dataset_dict[split].map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)
        
        print("Filtering the dataset...")
        dataset_dict[split] = filter_audio_length(dataset_dict[split], verbose=True)
        dataset_dict[split] = filter_labels(dataset_dict[split], min_nb_words=1, verbose=True)  # filter out empty labels
        
        print(f"Number of samples in the {split} set: {len(dataset_dict[split])}")
    
    return dataset_dict
