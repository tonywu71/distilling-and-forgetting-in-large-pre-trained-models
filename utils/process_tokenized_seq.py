from typing import Dict, Any
import torch


def remove_padding_fct(x: Dict[str, Any], 
                       col_sequences: str,
                       col_timestamps: str,
                       eot_token: int = 50257) -> Dict[str, Any]:
    """
    Remove padding added in the output of `model.generate()`.
    """
    sequences = x[col_sequences]
    timestamps = x[col_timestamps]

    count = (sequences == eot_token).sum()
    
    if count == 0 or count == 1:
        return {col_sequences: sequences, col_timestamps: timestamps}
    else:
        slice_idx = count - 1
        mask = torch.ones_like(sequences, dtype=torch.bool)
        mask[-slice_idx:] = 0
        return {col_sequences: sequences[mask], col_timestamps: timestamps[mask]}
