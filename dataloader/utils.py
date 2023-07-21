from typing import Any, Dict, Optional, Callable

import torch
from transformers import WhisperTokenizer, WhisperTokenizerFast
from datasets import Dataset


def get_fast_tokenizer(tokenizer: WhisperTokenizer) -> WhisperTokenizerFast:
    """
    IMPORTANT: There is a bug in the current version of transformers (4.30.2) that makes
               `WhisperTokenizerFast` not work properly as it won't output the special tokens
               for `language` and `task`.
    HOTFIX: Concatenate the special tokens to the vocabulary of the fast tokenizer manually
            using `add_missing_special_tokens_to_fast_tokenizer`.
    """
    return WhisperTokenizerFast.from_pretrained(tokenizer.name_or_path,
                                                language=tokenizer.language,
                                                task=tokenizer.task)


def get_map_funcion_to_restore_missing_special_tokens(col: str,
                                                      pretrained_model_name_or_path: str,
                                                      language: Optional[str] = None,
                                                      task: Optional[str] = None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Concatenate the language and task special tokens to the tokenized labels.
    Important: We assumed that all token sequences begin with a SOT token.
    """
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
    missing_tokens = tokenizer("").input_ids[1:3]
    
    def map_funcion_to_restore_missing_special_tokens(x: Dict[str, Any]) -> Dict[str, Any]:
        return {col: torch.cat([x[col][0:1], torch.LongTensor(missing_tokens), x[col][1:]])}
    
    return map_funcion_to_restore_missing_special_tokens


def remove_unnecessary_features_for_1_best(ds: Dataset, verbose: bool = True) -> Dataset:
    """
    Remove unnecessary features from the dataset to save memory. Used before saving 1-best cache.
    """
    COLS_FOR_1_BEST = ["audio", "text", "input_features", "input_ids", "labels", "teacher_sequences", "token_timestamps"]
    cols_to_remove = [feature_name for feature_name in ds.column_names if feature_name not in COLS_FOR_1_BEST]
    ds = ds.remove_columns(cols_to_remove)
    if verbose:
        print(f"Removed the following columns from the dataset: {cols_to_remove}")
    return ds


