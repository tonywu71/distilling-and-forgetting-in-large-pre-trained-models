from typing import Callable, Optional
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def get_whisper_normalizer(language: Optional[str] = None,
                           pretrained_model_name_or_path: Optional[str] = "openai/whisper-tiny.en") -> Callable[[str], str]:
    """If `language` is `None`, the basic Whisper normalizer is used."""
    if language in ["en", "english"]:
        whisper_norm = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path)._normalize
    else:
        whisper_norm = BasicTextNormalizer()
    return whisper_norm
