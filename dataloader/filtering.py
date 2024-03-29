from typing import Any, Dict
from functools import partial

from transformers.models.whisper import WhisperTokenizerFast
from datasets import Dataset, IterableDataset

from utils.distil_config import DistilConfig
from utils.whisper_hallucinations.get_features import count_zero_length_elements, get_audio_length_in_seconds, compute_gzip_compression_ratio
from utils.constants import (DEFAULT_LABEL_STR_COL,
                             MIN_INPUT_LENGTH,
                             MAX_INPUT_LENGTH,
                             DEFAULT_NUM_PROC)


def is_audio_in_length_range(audio: Dict[str, Any]) -> bool:
    """Return True if the audio is in the length range, False otherwise."""
    input_length = len(audio["array"]) / audio["sampling_rate"]  # type: ignore
    return MIN_INPUT_LENGTH < input_length < MAX_INPUT_LENGTH


def filter_audio_length(dataset: Dataset | IterableDataset,
                        verbose: bool = False) -> Dataset | IterableDataset:
    """
    Filter out audio examples that are not in the length range.
    """
    
    # Sanity check:
    assert "audio" in dataset.column_names, "Audio column not found in dataset."
    
    if isinstance(dataset, IterableDataset):
        dataset = dataset.filter(is_audio_in_length_range,
                                 input_columns=["audio"])
    else:
        n_rows_before = len(dataset)
        
        dataset = dataset.filter(is_audio_in_length_range,
                                input_columns=["audio"],
                                num_proc=DEFAULT_NUM_PROC)
        
        n_rows_after = len(dataset)
        
        if verbose:
            print(f"Removed {n_rows_before - n_rows_after} examples (audio that are not in the length range).")
    
    return dataset


def compute_label_length_fct(label: str) -> Dict[str, int]:
    """Return the length of the label in words."""
    return {"label_length": len(label.split())}


def filter_labels(dataset: Dataset | IterableDataset,
                  min_nb_words: int = 1,
                  label_col: str = DEFAULT_LABEL_STR_COL,
                  verbose: bool = False) -> Dataset | IterableDataset:
    """
    Filter out examples with stricly less than `min_nb_words` words in the label.
    """
    
    # Sanity checks:
    assert min_nb_words > 0, f"Minimum number of words must be positive, got {min_nb_words}."
    assert label_col in dataset.column_names, f"Label column '{label_col}' not found in dataset."
    
    if isinstance(dataset, IterableDataset):
        # Compute label length:
        dataset = dataset.map(partial(compute_label_length_fct),
                              input_columns=[label_col])
        
        # Filter out examples with stricly less than `min_nb_words` words:
        dataset = dataset.filter(lambda x: (x >= min_nb_words),
                                 input_columns=["label_length"])
        
        # Remove the label length column:
        dataset = dataset.remove_columns("label_length")
    
    else:
        n_rows_before = len(dataset)
        
        # Compute label length:
        dataset = dataset.map(partial(compute_label_length_fct),
                              input_columns=[label_col])
        
        # Filter out examples with stricly less than `min_nb_words` words:
        dataset = dataset.filter(lambda x: (x >= min_nb_words),
                                 input_columns=["label_length"],
                                 num_proc=DEFAULT_NUM_PROC)
        
        # Remove the label length column:
        dataset = dataset.remove_columns("label_length")
        
        n_rows_after = len(dataset)
        
        if verbose:
            print(f"Removed {n_rows_before - n_rows_after} examples (labels that contained stricly less than {min_nb_words} words).")
        
    return dataset


def filter_samples_1_best(ds: Dataset, config: DistilConfig) -> Dataset:
    """
    Filter out samples for which the teacher is not good enough for 1-best knowledge distillation.
    """

    tokenizer = WhisperTokenizerFast.from_pretrained(config.teacher_model_name_or_path, language=config.lang_name, task=config.task)

    print("Computing audio length...")
    ds = ds.map(get_audio_length_in_seconds, num_proc=DEFAULT_NUM_PROC)

    n_rows_before = ds.num_rows
    audio_length_before = sum(ds["audio_length"]) / 3600  # in hours
    print(f"Dataset before filtering: {n_rows_before} samples")
    print(f"Total audio length before filtering: {audio_length_before:.2f} hours")

    if config.max_exceeding_tokens:
        print(f"Filtering out samples where the teacher's text is longer than the student's labels + {config.max_exceeding_tokens} tokens...")
        ds = ds.filter(lambda x: len(x["teacher_sequences"]) - len(x["labels"]) <= config.max_exceeding_tokens, num_proc=DEFAULT_NUM_PROC)

    if config.max_teacher_gzip_ratio:
        print(f"Filtering out samples with a teacher compression ratio greater than {config.max_teacher_gzip_ratio}...")
        def filter_teacher_gzip_ratio(x: Dict[str, Any]) -> bool:
            teacher_seq = tokenizer.decode(x["teacher_sequences"], skip_special_tokens=True)
            return (compute_gzip_compression_ratio(teacher_seq) <= config.max_teacher_gzip_ratio)
        ds = ds.filter(filter_teacher_gzip_ratio, num_proc=DEFAULT_NUM_PROC)
    
    if config.max_ratio_instant_tokens:
        print(f"Filtering out samples with a ratio of instant tokens greater than {config.max_ratio_instant_tokens}...")
        ds = ds.map(lambda x: {"ratio_instant_tokens": count_zero_length_elements(x["token_timestamps"]) / len(x["teacher_sequences"])},
                    num_proc=DEFAULT_NUM_PROC)
        ds = ds.filter(lambda x: x["ratio_instant_tokens"] <= config.max_ratio_instant_tokens, num_proc=DEFAULT_NUM_PROC)
    
    n_rows_after = ds.num_rows
    audio_length_after = sum(ds["audio_length"]) / 3600 # in hours
    print(f"Dataset after filtering: {n_rows_after} samples")
    print(f"Total audio length after filtering: {audio_length_after:.2f} hours")

    print(f"Filtered out {n_rows_before - n_rows_after} samples ({(n_rows_before - n_rows_after) / n_rows_before * 100:.2f}%)")
    print(f"Filtered out {audio_length_before - audio_length_after:.2f} hours of audio ({(audio_length_before - audio_length_after) / audio_length_before * 100:.2f}%)")

    ds = ds.remove_columns(["audio_length"])

    return ds
