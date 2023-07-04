from typing import Callable, Optional
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def get_whisper_normalizer(language: Optional[str]=None) -> Callable[[str], str]:
    """If `language` is `None`, the basic Whisper normalizer is used."""
    if language in ["en", "english"]:
        whisper_norm = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")._normalize
    else:
        whisper_norm = BasicTextNormalizer()
    return whisper_norm
