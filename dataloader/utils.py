from transformers import WhisperTokenizer, WhisperTokenizerFast


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
