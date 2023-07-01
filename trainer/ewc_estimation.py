from typing import Dict, Tuple
from tqdm.auto import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

from transformers import PreTrainedModel
from transformers.models.whisper import (WhisperTokenizerFast,
                                         WhisperFeatureExtractor,
                                         WhisperForConditionalGeneration)
from transformers.models.whisper import WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataset_loader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import prepare_dataset_fct
from utils.constants import DEFAULT_NUM_PROC


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
    
    list_sum_fisher_params = [torch.zeros_like(param) for param in model.parameters()]
    total = len(dataloader)
    
    for inputs in tqdm(dataloader, total=total):
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        log_prob_all = torch.nn.functional.log_softmax(outputs.logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        log_prob = log_prob_all.take_along_dim(inputs.labels[..., None], dim=-1)  # (batch_size, seq_len, 1)
        log_prob = log_prob.squeeze().sum(dim=-1)  # (batch_size,)
        log_likelihood = torch.sum(log_prob)  # summation as we compute the prob of independent RV in log-space -> (1,)
        grad_log_likelihood = torch.autograd.grad(log_likelihood, model.parameters())  # Tuple of P tensors (each tensor is a model param) where P=#params
        
        for idx, (fisher_param, grad_param) in enumerate(zip(list_sum_fisher_params, grad_log_likelihood)):
            list_sum_fisher_params[idx] = fisher_param + grad_param.clone() ** 2
    
    fisher_params = {param_name: (fisher_param / total) for (param_name, param), fisher_param in zip(model.named_parameters(), list_sum_fisher_params)}
    return fisher_params


def get_ewc_params(model: PreTrainedModel,
                   dataloader: DataLoader) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns the EWC parameters of the model.
    """
    mean_params = get_mean_params(model)
    fisher_params = get_fisher_params(model, dataloader)
    return mean_params, fisher_params


def get_ewc_params_for_whisper(pretrained_model_name_or_path: str,
                               language: str,
                               task: str,
                               dataset_name: str,
                               batch_size: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns the EWC parameters for a pretrained Whisper model.
    """
    
    # Get the device:
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():  # for Apple Silicon
        device = torch.device('mps')
    else:
        device = "cpu"
    
    # Initialize the model from a pretrained checkpoint:
    print(f"Loading pretrained model from `{pretrained_model_name_or_path}`...")
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path).to(device)
    if device == "cuda:0":
        model = BetterTransformer.transform(model)
    
    # Initialize the tokenizer and feature extractor:
    tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    
    # Load and prepare the dataset:
    ds = load_dataset_dict(dataset_name)["validation"]
    prepare_dataset = partial(prepare_dataset_fct,
                              tokenizer=tokenizer,
                              feature_extractor=feature_extractor)
    ds = ds.map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)
    
    # Get the dataloader:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                         feature_extractor=feature_extractor,
                                                         return_attention_mask=True,
                                                         replace_padded_with_loss_mask_for_labels=False,
                                                         discard_first_bos_token=True)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
    
    # Get the EWC params:
    mean_params, fisher_params = get_ewc_params(model, dataloader)
    
    return mean_params, fisher_params
