from typing import Dict, Tuple
from pathlib import Path

import torch
from safetensors import safe_open


def load_ewc_params(dirpath_ewc: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load the EWC parameters from the given directory path.
    """
    filepath_ewc_mean = Path(dirpath_ewc) / "ewc_mean_params.safetensors"
    assert filepath_ewc_mean.exists(), f"`filepath_ewc_mean` does not exist: {filepath_ewc_mean}"
    
    filepath_ewc_fisher = Path(dirpath_ewc) / "ewc_fisher_params.safetensors"
    assert filepath_ewc_fisher.exists(), f"`filepath_ewc_fisher` does not exist: {filepath_ewc_fisher}"
    
    ewc_mean_params = {}
    with safe_open(filepath_ewc_mean, framework="pt", device=0) as f:
        for k in f.keys():
            ewc_mean_params[k] = f.get_tensor(k)
    
    ewc_fisher_params = {}
    with safe_open(filepath_ewc_fisher, framework="pt", device=0) as f:
        for k in f.keys():
            ewc_fisher_params[k] = f.get_tensor(k)
    
    return ewc_mean_params, ewc_fisher_params
