from transformers import WhisperTokenizer, WhisperTokenizerFast


def get_fast_tokenizer_from_tokenizer(tokenizer: WhisperTokenizer) -> WhisperTokenizerFast:
    return WhisperTokenizerFast.from_pretrained(tokenizer.name_or_path,
                                                language=tokenizer.language,
                                                task=tokenizer.task)
