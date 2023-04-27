from typing import Callable
from transformers import WhisperTokenizer


def get_whisper_normalizer(whisper_tokenizer: WhisperTokenizer) -> Callable:
    """Get the normalization function from the whisper_asr pipeline."""
    whisper_norm = whisper_tokenizer._normalize  # type: ignore
    return whisper_norm
