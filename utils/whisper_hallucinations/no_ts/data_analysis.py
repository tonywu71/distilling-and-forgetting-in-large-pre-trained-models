from typing import Dict, Any

import pandas as pd
from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from datasets import Dataset


def get_audio_length_in_seconds(x: Dict[str, Any]) -> Dict[str, float]:
    assert "audio" in x, "x must have an 'audio' key"
    audio = x["audio"]
    audio_length = len(audio["array"]) / audio["sampling_rate"]
    return {"audio_length": audio_length}


def add_features_to_ds(ds: Dataset,
                       tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                       num_proc: int = 1) -> Dataset:
    assert "pred" in ds.features, "ds must have a 'pred' feature"

    # Add audio length to the dataset features:
    ds = ds.map(get_audio_length_in_seconds, num_proc=num_proc)

    # Tokenize both labels and predictions:
    ds = ds.map(lambda batch: {"labels": tokenizer(batch["text"]).input_ids,
                               "pred_tokenized": tokenizer(batch["pred"]).input_ids},
                batched=True)
    
    # Add n_tokens to the dataset features:
    ds = ds.map(lambda x: {"n_tokens_labels": len(x["labels"]), "n_tokens_pred": len(x["pred_tokenized"])})

    return ds


def get_df_from_ds(ds: Dataset) -> pd.DataFrame:
    cols_of_interest = ["audio_length", "text", "labels", "n_tokens_labels", "pred", "pred_tokenized", "n_tokens_pred"]
    df = pd.DataFrame({col: ds[col] for col in cols_of_interest})
    return df


def add_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diff_n_tokens"] = df["n_tokens_pred"] - df["n_tokens_labels"]
    return df
