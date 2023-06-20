from typing import Any, Dict
from functools import partial
from datasets import Dataset

from utils.constants import (DEFAULT_LABEL_STR_COL,
                             MIN_INPUT_LENGTH,
                             MAX_INPUT_LENGTH,
                             DEFAULT_NUM_PROC)


def is_audio_in_length_range(audio: Dict[str, Any]) -> bool:
    """Return True if the audio is in the length range, False otherwise."""
    input_length = len(audio["array"]) / audio["sampling_rate"]  # type: ignore
    return MIN_INPUT_LENGTH < input_length < MAX_INPUT_LENGTH


def filter_audio_length(dataset: Dataset,
                        verbose: bool = False) -> Dataset:
    """
    Filter out audio examples that are not in the length range.
    """
    
    # Sanity check:
    assert "audio" in dataset.column_names, "Audio column not found in dataset."
    
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


def filter_labels(dataset: Dataset,
                  min_nb_words: int = 1,
                  label_col: str = DEFAULT_LABEL_STR_COL,
                  verbose: bool = False) -> Dataset:
    """
    Filter out examples with stricly less than `min_nb_words` words in the label.
    """
    
    # Sanity checks:
    assert min_nb_words > 0, f"Minimum number of words must be positive, got {min_nb_words}."
    assert label_col in dataset.column_names, f"Label column '{label_col}' not found in dataset."
    
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
