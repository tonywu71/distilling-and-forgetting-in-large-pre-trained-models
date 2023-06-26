from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

def get_language_token(language: str):
    """
    Get the Whisper language token for a given (supported) language.
    """
    idx_shift = 50258
    langs = tuple(LANGUAGES.keys())
    
    language = language.lower()
    if language in TO_LANGUAGE_CODE:
        language_id = TO_LANGUAGE_CODE[language]
    elif language in TO_LANGUAGE_CODE.values():
        language_id = language
    else:
        is_language_code = len(language) == 2
        raise ValueError(
            f"Unsupported language: {language}. Language should be one of:"
            f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
        )
    
    return idx_shift + 1 + langs.index(language_id)
