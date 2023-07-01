from typing import Dict
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers.models.whisper import WhisperForConditionalGeneration


def get_mean_params(model: WhisperForConditionalGeneration) -> Dict[str, torch.Tensor]:
    """
    Returns the mean parameters of the model.
    """
    mean_params = {param_name: param.clone() for param_name, param in model.named_parameters()}
    return mean_params


def get_fisher_params(model: WhisperForConditionalGeneration,
                      dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Returns the EWC parameters of the model.
    """
    
    log_liklihoods = []
    
    for inputs in tqdm(dataloader, total=len(dataloader)):
        outputs = model(**inputs)
        log_prob_all = torch.nn.functional.log_softmax(outputs.logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        log_prob = log_prob_all.take_along_dim(outputs.labels[..., None], dim=-1)  # (batch_size, seq_len, 1)
        log_prob = log_prob.squeeze().sum(dim=-1)  # (batch_size,)
        log_likelihood = torch.sum(log_prob)  # summation as we compute the prob of independent RV in log-space -> (1,)
        log_liklihoods.append(log_likelihood)
    
    log_likelihood = torch.cat(log_liklihoods).mean()  # (1,)
    
    grad_log_likelihood = torch.autograd.grad(log_likelihood, model.parameters())
    
    fisher_params = {param_name: grad_param.clone() ** 2 for (param_name, param), grad_param in zip(model.named_parameters(), grad_log_likelihood)}
    return fisher_params
