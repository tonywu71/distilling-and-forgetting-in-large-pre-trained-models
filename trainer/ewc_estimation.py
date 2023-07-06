from typing import Dict, Tuple, Optional
import os
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import (WhisperTokenizer,
                                         WhisperFeatureExtractor,
                                         WhisperForConditionalGeneration)
from transformers.models.whisper import WhisperForConditionalGeneration
from datasets import Dataset, load_from_disk

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataset_loader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import lowercase_fct, prepare_dataset_fct
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


def load_and_prepare_dataset(dataset_name: str,
                             split: str,
                             tokenizer: WhisperTokenizer,
                             feature_extractor: WhisperFeatureExtractor,
                             lowercase: bool = True) -> Dataset:
    """
    Load and prepare the dataset.
    """
    ds = load_dataset_dict(dataset_name)[split]
    prepare_dataset = partial(prepare_dataset_fct,
                              tokenizer=tokenizer,
                              feature_extractor=feature_extractor)
    if lowercase:
        ds = ds.map(lowercase_fct, num_proc=DEFAULT_NUM_PROC)
    ds = ds.map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)
    return ds


def get_ewc_params_for_whisper(pretrained_model_name_or_path: str,
                               language: str,
                               task: str,
                               dataset_name: str,
                               split: str = "train",
                               batch_size: int = 32,
                               lowercase: bool = True,
                               cache_dir: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns the EWC parameters for a pretrained Whisper model.
    """
    
    assert split in ["train", "validation"], f"Invalid `split` value: {split}"
    
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
    
    # Initialize the tokenizer and feature extractor:
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    
    # Load and prepare the dataset:
    if cache_dir:
        if os.path.isdir(cache_dir):
            print(f"Found cache at `{cache_dir}`. Loading pre-processed dataset...")
            ds = load_from_disk(cache_dir)
            print("Successfully loaded pre-processed dataset.")
        else:
            print(f"Cache not found at `{cache_dir}`. The dataset will be pre-processed and cached for future use.")
            ds = load_and_prepare_dataset(dataset_name=dataset_name,
                                          split=split,
                                          tokenizer=tokenizer,
                                          feature_extractor=feature_extractor,
                                          lowercase=lowercase)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(cache_dir)
            print(f"Dataset pre-processed and cached at `{cache_dir}`.")
    else:
        print("No cache directory provided. The dataset will be pre-processed but not cached.")
        ds = load_and_prepare_dataset(dataset_name=dataset_name,
                                      split=split,
                                      tokenizer=tokenizer,
                                      feature_extractor=feature_extractor,
                                      lowercase=lowercase)
        print("Dataset pre-processed.")
    
    # Get the dataloader:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                         feature_extractor=feature_extractor,
                                                         return_attention_mask=True,
                                                         replace_padded_with_loss_mask_for_labels=False,
                                                         discard_first_bos_token=True)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
    
    # Get the EWC params:
    print("Computing the EWC parameters...")
    mean_params, fisher_params = get_ewc_params(model, dataloader)
    
    return mean_params, fisher_params
