import torch

def remove_redundant_eot(tensor: torch.Tensor, eot_token: int = 50257) -> torch.Tensor:
    """
    Remove redundant EOT tokens from a tensor.
    Note that we need this function because the EOT token is the same as the PAD token.
    """
    count = (tensor == eot_token).sum()
    if count == 0 or count == 1:
        return tensor
    else:
        slice_idx = count - 1
        mask = torch.ones_like(tensor, dtype=torch.bool)
        mask[-slice_idx:] = 0
        return tensor[mask]
