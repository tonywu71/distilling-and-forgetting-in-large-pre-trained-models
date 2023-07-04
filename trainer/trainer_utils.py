import torch
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


def get_padded_mask_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the padded mask from a tensor of shape (batch_size, n_tokens).
    Used convention:
    - 1 for tokens that are padded
    - 0 otherwise.
    
    Example:
    - Input: tensor([[50257.,  50362.,     76.,    1694.,    627.,   50256.],
                        [50257.,  50362.,  13099.,   50256.,  50256.,   50256.]])
    - Output: tensor([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1]])
    """
    PAD_TOKEN_FROM_GENERATE = 50257  # different from the one used in the tokenizer
    assert tensor.ndim == 2, \
        f"The tensor must be 2D. Got {tensor.ndim} dimensions."
    
    indices = (tensor == PAD_TOKEN_FROM_GENERATE).long().argmax(dim=-1)
    padded_mask = torch.zeros_like(tensor, dtype=torch.long)
    for idx, row in zip(indices, padded_mask):
        row[idx+1:] = 1  # ignore the first EOT token of each row
    return padded_mask
