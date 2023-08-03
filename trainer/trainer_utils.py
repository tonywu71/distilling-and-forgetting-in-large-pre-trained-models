import torch
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE


def get_language_special_token(language: str) -> int:
    """
    Get the Whisper language token for a given (supported) language.
    IMPORTANT: Only applies to the multilingual Whisper tokenizer.
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


def get_task_special_token(task: str) -> int:
    """
    Get the Whisper task token for a given task.
    IMPORTANT: Only applies to the multilingual Whisper tokenizer.
    """
    TRANSLATE_TOKEN_ID = 50358
    TRANSCRIBE_TOKEN_ID = 50359
    if task == "translate":
        return TRANSLATE_TOKEN_ID
    elif task == "transcribe":
        return TRANSCRIBE_TOKEN_ID
    else:
        raise ValueError(f"Unsupported task: {task}. Task should be one of: ['translate', 'transcribe'].")


def get_padded_mask_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the padded mask from a tensor of shape (batch_size, n_tokens).
    Used convention:
    - 0 for tokens that are padded
    - 1 otherwise.
    
    Example:
    - Input: tensor([[50257.,  50362.,     76.,    1694.,    627.,   50256.],
                        [50257.,  50362.,  13099.,   50256.,  50256.,   50256.]])
    - Output: tensor([[1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 0]])
    """
    PAD_TOKEN_FROM_GENERATE = 50257  # different from the one used in the tokenizer
    assert tensor.ndim == 2, \
        f"The tensor must be 2D. Got {tensor.ndim} dimensions."
    
    indices = (tensor == PAD_TOKEN_FROM_GENERATE).long().argmax(dim=-1)
    padded_mask = torch.ones_like(tensor, dtype=torch.long)
    for idx, row in zip(indices, padded_mask):
        row[idx+1:] = 0  # ignore the first EOT token of each row
    return padded_mask
