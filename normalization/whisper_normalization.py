from typing import Callable
from transformers import Pipeline


def get_whisper_normalizer(whisper_asr: Pipeline) -> Callable:
    """Get the normalization function from the whisper_asr pipeline."""
    whisper_norm = whisper_asr.tokenizer._normalize  # type: ignore
    return whisper_norm
