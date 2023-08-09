import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import numpy as np
import torch

import pandas as pd
from tqdm.auto import tqdm

from transformers.models.whisper import (WhisperTokenizer,
                                         WhisperTokenizerFast,
                                         WhisperFeatureExtractor)
from optimum.bettertransformer import BetterTransformer

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from models.whisper_zero_cross_attention import WhisperForConditionalGenerationZeroCrossAttention
from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, DEFAULT_LABEL_STR_COL, DEFAULT_EVAL_BATCH_SIZE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_whisper_implicit_lm_on_dataset_group(pretrained_model_name_or_path: str,
                                              ds_group: BaseDatasetGroup,
                                              batch_size: int = DEFAULT_EVAL_BATCH_SIZE,  # only 1 is supported for now
                                              fast_tokenizer: bool = True,
                                              task: str="transcribe") -> pd.Series:
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16  # see https://huggingface.co/learn/audio-course/chapter5/evaluation?fw=pt
    elif torch.backends.mps.is_available():  # for Apple Silicon
        device = torch.device('mps')
        torch_dtype = torch.float32  # float16 not supported by MPS
    else:
        device = "cpu"
        torch_dtype = torch.float32

    # Load model:
    model = WhisperForConditionalGenerationZeroCrossAttention.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype).to(device)
    
    if device == "cuda:0":
        model = BetterTransformer.transform(model)

    # Loop over the datasets:
    perplexity_results = []
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Evaluating {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        if fast_tokenizer:
            tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        else:
            tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        
        # Load data collator:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                             feature_extractor=feature_extractor,
                                                             return_attention_mask=True)
        
        # TODO: Currently, perplexity is computed with a batch size of 1. For faster inference, we should
        #       add support for batch sizes > 1. The only thing to take care of is the padding tokens.

        # Placeholders for per-example perplexities:
        perplexities_curr_dataset = []
        
        for data in dataset:
            data = {
                "input_features": feature_extractor(data["audio"]["array"],
                                                    sampling_rate=feature_extractor.sampling_rate).input_features[0],  # drop batch dimension
                DEFAULT_LABEL_TOKENIZED_COL: tokenizer(data[DEFAULT_LABEL_STR_COL]).input_ids
            }
            
            # Collate the data into batches of size 1:
            data = data_collator([data])
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device).to(torch_dtype)
            tokenized_seq = concat_special_tokens(data[DEFAULT_LABEL_TOKENIZED_COL].to(device),
                                                  pretrained_model_name_or_path,
                                                  language=language,
                                                  task=task)
            attention_mask = data["attention_mask"].to(device)
            attention_mask_prefix = torch.Tensor([1, 1, 1, 1]).expand(attention_mask.shape[0], -1).to(attention_mask.device)
            attention_mask = torch.cat([attention_mask_prefix, attention_mask[:, 2:]], dim=1)

            # Shift inputs for next-word prediction:
            decoder_input_ids = tokenized_seq[:, 1:]
            decoder_input_ids_right_shifted = tokenized_seq[:, :-1]
            attention_mask_right_shifted = attention_mask[:, :-1]

            # One-step generation:
            with torch.no_grad():
                output = model.forward(input_features=input_features,
                                    decoder_input_ids=decoder_input_ids_right_shifted,
                                    attention_mask=attention_mask_right_shifted)
            
            # Convert logits to log-probabilities:
            log_prob_all = torch.nn.functional.log_softmax(output.logits, dim=-1)  # (batch_size, seq_len, vocab_size)

            # Take probabilities for the ground-truth tokens:
            log_prob_tokens = log_prob_all.take_along_dim(decoder_input_ids[..., None], dim=-1).squeeze(dim=-1)  # (batch_size, seq_len)
            
            # All the values associated to the pad tokens will be set to 0 in order to ignore them when we will sum.
            log_prob_seq = log_prob_tokens.masked_fill(attention_mask_right_shifted.eq(0), 0).sum(dim=-1)  # (batch_size,)
            mean_log_prob_seq = log_prob_seq / attention_mask_right_shifted.sum(dim=-1)  # (batch_size,)
            
            # Compute perplexity:
            perplexity = torch.exp(-mean_log_prob_seq).item()
            
            # Add to the list of perplexities:
            perplexities_curr_dataset.append(perplexity)
        
        # Add to the list of perplexities:
        perplexity_results.append(np.mean(perplexities_curr_dataset))
    
    
    # Save the results:
    results = pd.Series(perplexity_results, index=list(ds_group.keys()), name="Perplexity")
    results.index.name = "Dataset"
    
    # Compute the average WER:
    results["Average"] = results.mean()
    
    # Round the results:
    results = results.round(2)
    
    return results


def concat_special_tokens(x: torch.Tensor,
                          pretrained_model_name_or_path: str,
                          language: Optional[str] = None,
                          task: Optional[str] = None) -> torch.Tensor:
    """
    Concatenate the language and task special tokens to the tokenized labels (batched).
    Important: We assumed that all token sequences begin with `<sot>, <notimestamp>`.
    """
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
    special_tokens = torch.LongTensor([tokenizer("").input_ids[:4]]).expand(x.shape[0], -1).to(x.device)
    x = torch.cat([special_tokens, x[:, 2:]], dim=1)
    return x
