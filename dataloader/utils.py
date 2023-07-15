from transformers import WhisperTokenizer, WhisperTokenizerFast
from datasets import Dataset


def get_fast_tokenizer(tokenizer: WhisperTokenizer) -> WhisperTokenizerFast:
    """
    IMPORTANT: There is a bug in the current version of transformers (4.30.2) that makes
               `WhisperTokenizerFast` not work properly. It would forget to output the special tokens
               for `language` and `task`.
    HOTFIX: Concatenate the special tokens to the vocabulary of the fast tokenizer manually.
    """
    return WhisperTokenizerFast.from_pretrained(tokenizer.name_or_path,
                                                language=tokenizer.language,
                                                task=tokenizer.task)


def remove_unnecessary_features_for_1_best(ds: Dataset, verbose: bool = True) -> Dataset:
    """
    Remove unnecessary features from the dataset to save memory. Used before saving 1-best cache.
    """
    COLS_FOR_1_BEST = ["audio", "text", "input_features", "input_ids", "labels", "teacher_sequences"]
    cols_to_remove = [feature_name for feature_name in ds.column_names if feature_name not in COLS_FOR_1_BEST]
    ds = ds.remove_columns(cols_to_remove)
    if verbose:
        print(f"Removed the following columns from the dataset: {cols_to_remove}")
    return ds
