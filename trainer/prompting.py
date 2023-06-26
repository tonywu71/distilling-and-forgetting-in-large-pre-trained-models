from typing import Tuple
import torch
from transformers import WhisperTokenizer


BOS_TOKEN_ID = 50258


def get_labels_with_prompt(labels: torch.Tensor,
                           tokenizer: WhisperTokenizer,
                           language: str = "en",
                           task: str = "transcribe",
                           no_timestamps: bool = True)-> Tuple[torch.Tensor, int, int]:
    """
    --- LEGACY: kept for reference ---
    
    Returns the labels with the prefix and suffix tokens, as well as the number of prefix and suffix tokens.
    `labels_with_prompt` should be used as the `decoder_input_ids` argument for the `forward` method of the model.
    
    Note: n_prefix_tokens should be 4 (BOS, language, task, if_timestamps) and n_suffix_tokens should be 1 (EOS).
    """
    
    # Get batch size:
    batch_size = labels.shape[0]

    # Get prefix tokens:
    forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=language, task=task, no_timestamps=no_timestamps)  # language, task, if_timestamps
    prefix_tokens = torch.IntTensor([BOS_TOKEN_ID] + [token_id for idx, token_id in forced_decoder_ids])  # (n_prefix_tokens, )
    prefix_tokens = prefix_tokens.expand(batch_size, -1)  # (batch_size, n_prefix_tokens)

    # Get suffix tokens:
    suffix_tokens = torch.IntTensor([tokenizer.eos_token_id])  # (n_suffix_tokens, )
    suffix_tokens = suffix_tokens.expand(batch_size, -1)  # (batch_size, n_suffix_tokens)

    # Get prefix and suffix lengths:
    n_prefix_tokens = prefix_tokens.shape[1]  # n_prefix_tokens
    n_suffix_tokens = suffix_tokens.shape[1]  # n_suffix_tokens
    
    # Send tensors to the same device as the `labels` tensor:
    prefix_tokens = prefix_tokens.to(labels.device)
    suffix_tokens = suffix_tokens.to(labels.device)
    
    # Concatenate the prefix tensor with the original tensor along the second dimension:
    labels_with_prompt = torch.cat((prefix_tokens, labels, suffix_tokens), dim=1)  # (batch_size, n_tokens_labels + n_prefix_tokens + n_suffix_tokens)

    return labels_with_prompt, n_prefix_tokens, n_suffix_tokens


def get_attention_mask_with_prompt(attention_mask_labels: torch.Tensor,
                                   n_prefix_tokens: int,
                                   n_suffix_tokens: int) -> torch.Tensor:
    """
    --- LEGACY: kept for reference ---
    
    Returns the attention mask for which the correct mask was added for the prefix and suffix tokens.
    """
    
    # Get batch size:
    batch_size = attention_mask_labels.shape[0]
    
    # Get prefix and suffix attention masks:
    attention_prefix = torch.ones(batch_size, n_prefix_tokens)  # (batch_size, n_prefix_tokens)
    attention_suffix = torch.ones(batch_size, n_suffix_tokens)  # (batch_size, n_suffix_tokens)
    
    # Send tensors to the same device as the `labels` tensor:
    attention_prefix = attention_prefix.to(attention_mask_labels.device)
    attention_suffix = attention_suffix.to(attention_mask_labels.device)
    
    attention_mask_labels_with_prompt = torch.cat([attention_prefix, attention_mask_labels, attention_suffix], dim=1)  # (batch_size, n_tokens_labels + n_prefix_tokens + n_suffix_tokens)
    
    return attention_mask_labels_with_prompt
