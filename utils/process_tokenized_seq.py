from typing import Dict, Any, Optional
import torch


def remove_padding_fct(x: Dict[str, Any], 
                       col_sequences: str,
                       col_timestamps: Optional[str] = None,
                       eot_token: int = 50257) -> Dict[str, Any]:
    """
    Remove padding added in the output of `model.generate()`.
    """
    sequences = x[col_sequences]
    if col_timestamps:
        timestamps = x[col_timestamps]

    count = (sequences == eot_token).sum()
    
    if count == 0 or count == 1:
        outputs = {col_sequences: sequences}
        if col_timestamps:
            outputs[col_timestamps] = timestamps
        return outputs
    else:
        slice_idx = count - 1
        mask = torch.ones_like(sequences, dtype=torch.bool)
        mask[-slice_idx:] = 0
        outputs = {col_sequences: sequences[mask]}
        if col_timestamps:
            outputs[col_timestamps] = timestamps[mask]
        return outputs


def remove_padding_k_beam(x: torch.Tensor, eot_token: int = 50257) -> torch.Tensor:
    """
    Remove padding added in the output of `model.generate()` when used with k-beam search.
    """
    assert x.dim() == 2, f"Invalid shape: {x.shape}. Must be (batch_size, n_tokens)."

    count = (x == eot_token).sum(axis=1).min()
    
    if count == 0 or count == 1:
        return x
    
    slice_idx = count - 1
    mask = torch.ones_like(x[0], dtype=torch.bool)
    mask[-slice_idx:] = 0

    return x[:, mask]
