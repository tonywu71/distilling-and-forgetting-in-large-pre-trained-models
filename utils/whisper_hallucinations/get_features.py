from typing import Dict, Any

import pandas as pd
from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from datasets import Dataset


def get_audio_length_in_seconds(x: Dict[str, Any]) -> Dict[str, float]:
    assert "audio" in x, "x must have an 'audio' key"
    audio = x["audio"]
    audio_length = len(audio["array"]) / audio["sampling_rate"]
    return {"audio_length": audio_length}


def count_overlaps(result: Dict[str, Any]) -> int:
    counter = 0
    for segment in result["segments"]:
        for w1, w2 in zip(segment["words"], segment["words"][1:]):
            if w1["end"] > w2["start"]:
                counter += 1
    return counter


def check_words_within_delta(result, delta: float = 0.1, n_words: int = 5) -> bool:
    for segment in result['segments']:
        words = segment['words']
        for i in range(len(words) - n_words):
            if words[i + n_words]['start'] - words[i]['end'] <= delta:
                return True
    return False


def add_features_to_ds(ds: Dataset,
                       results: Dict[str, Any],
                       tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                       num_proc: int = 1) -> Dataset:
    
    assert "labels" in ds.features, "The dataset must have a 'labels' column for the tokenized ground-truth text"
    
    # Get teacher predictions:
    teacher_text = [x["text"].lower() for x in results]

    # Add teacher predictions to the dataset features:
    assert ds.num_rows == len(teacher_text), "Number of predictions must match number of examples in the dataset"
    ds = ds.add_column(name="teacher_text", column=teacher_text)

    # Tokenize teacher predictions:
    ds = ds.map(lambda batch: {"teacher_labels": tokenizer(batch["teacher_text"]).input_ids}, batched=True)

    # Add audio length to the dataset features:
    ds = ds.map(get_audio_length_in_seconds, num_proc=num_proc)

    # Add n_tokens to the dataset features:
    ds = ds.map(lambda x: {"n_tokens_labels": len(x["labels"]), "n_tokens_teacher": len(x["teacher_labels"])})

    # Add number of overlaps per example:
    ds = ds.add_column(name="n_overlaps", column=[count_overlaps(result) for result in results])

    # Add if the prediction is a fast utterance:
    ds = ds.add_column(name="is_fast_utterance", column=[check_words_within_delta(result) for result in results])

    # Add diff_n_tokens to the dataset features:
    ds = ds.map(lambda x: {"diff_n_tokens": x["n_tokens_teacher"] - x["n_tokens_labels"]})

    # Add max_token_repetitions to the dataset features:
    ds = ds.map(lambda x: {"max_token_repetitions_labels": pd.Series(x["labels"]).value_counts().max()})
    ds = ds.map(lambda x: {"max_token_repetitions_teacher": pd.Series(x["teacher_labels"]).value_counts().max()})

    return ds
