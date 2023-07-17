from typing import Dict, Any
import io
import gzip
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


def count_overlaps(result: Dict[str, Any]) -> int:
    counter = 0
    list_words = []
    for segment in result["segments"]:
        list_words.extend(segment["words"])
    for w1, w2 in zip(list_words, list_words[1:]):
        if w1["end"] > w2["start"]:
            counter += 1
    return counter



def add_features_to_ds(ds: Dataset,
                       results: Dict[str, Any],
                       tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                       num_proc: int = 1,
                       lowercase_teacher: bool = False) -> Dataset:
    
    assert "labels" in ds.features, "The dataset must have a 'labels' column for the tokenized ground-truth text"
    
    # Get teacher predictions:
    if lowercase_teacher:
        teacher_text = [x["text"].lower() for x in results]
    else:
        teacher_text = [x["text"] for x in results]
    
    # NOTE: When alpha_ce = 0, we shouldn't lowercase the teacher predictions because the goal of 1-best KD is for
    #       the student to learn to predict the raw teacher's predictions (without any normalization).

    # Add teacher predictions to the dataset features:
    assert ds.num_rows == len(teacher_text), "Number of predictions must match number of examples in the dataset"
    ds = ds.add_column(name="teacher_text", column=teacher_text)

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
    ds = ds.add_column(name="n_overlaps", column=[count_overlaps(result) for result in results])

    return ds
