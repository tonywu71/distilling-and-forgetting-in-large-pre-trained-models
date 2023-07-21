from typing import Dict, Any
import io
import gzip

import torch
from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from datasets import Dataset


def get_audio_length_in_seconds(x: Dict[str, Any]) -> Dict[str, float]:
    assert "audio" in x, "x must have an 'audio' key"
    audio = x["audio"]
    audio_length = len(audio["array"]) / audio["sampling_rate"]
    return {"audio_length": audio_length}


def compute_gzip_compression_ratio(text: str) -> float:
    # Convert text to bytes
    text_bytes = text.encode('utf-8')

    # Compress the bytes using gzip
    compressed_bytes = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_bytes, mode='wb') as f:
        f.write(text_bytes)

    # Compute the compression ratio
    compression_ratio = len(text_bytes) / len(compressed_bytes.getvalue())

    return compression_ratio


def count_zero_length_elements(tensor):
    end_times = torch.roll(tensor, -1)
    return torch.sum(end_times == tensor)


def max_subarray_length(x):
    # Compute the differences between adjacent elements
    diffs = torch.diff(x)

    # Find the indices where the differences are non-zero
    indices = torch.nonzero(diffs)

    # Compute the lengths of the subarrays between the indices
    lengths = torch.diff(torch.cat([torch.tensor([-1]), indices.flatten(), torch.tensor([len(x)])]))

    # Find the maximum length of any subarray with one unique value
    # max_length = torch.max(lengths[x[indices[:, 0]] == x[indices[:, 0] + 1]])
    max_length = torch.max(lengths)

    return max_length


def add_features_to_ds(ds: Dataset,
                       tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                       num_proc: int = 1) -> Dataset:
    """
    Add features to the dataset.
    """
    
    # Tokenize teacher predictions:
    ds = ds.map(lambda batch: {"teacher_labels": tokenizer(batch["teacher_text"]).input_ids}, batched=True)

    # Add audio length to the dataset features:
    ds = ds.map(get_audio_length_in_seconds, num_proc=num_proc)

    # Add n_tokens to the dataset features:
    ds = ds.map(lambda x: {"n_tokens_labels": len(x["labels"]), "n_tokens_teacher": len(x["teacher_labels"])})

    # Add diff_n_tokens to the dataset features:
    ds = ds.map(lambda x: {"diff_n_tokens": x["n_tokens_teacher"] - x["n_tokens_labels"]})

    # Add compression ratios to the dataset features:
    ds = ds.map(lambda x: {"gzip_ratio": compute_gzip_compression_ratio(x["text"]),
                           "teacher_gzip_ratio": compute_gzip_compression_ratio(x["teacher_text"])})
    
    # Add diff_gzip_ratio to the dataset features:
    ds = ds.map(lambda x: {"diff_gzip_ratio": x["teacher_gzip_ratio"] - x["gzip_ratio"]})

    # Add number of overlaps per example:
    ds = ds.map(lambda x: {"n_instant_tokens": count_zero_length_elements(x["token_timestamps"]),
                           "max_subarray_length": max_subarray_length(x["token_timestamps"])},
                           num_proc=num_proc)

    return ds
