import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List
from functools import partial

import torch
from torch.utils.data import DataLoader

import pandas as pd
from tqdm.auto import tqdm

from transformers.models.whisper import (WhisperTokenizer,
                                         WhisperTokenizerFast,
                                         WhisperFeatureExtractor)
from optimum.bettertransformer import BetterTransformer

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.preprocessing_train.preprocessing import prepare_dataset_fct
from models.whisper_zero_cross_attention import WhisperForConditionalGenerationZeroCrossAttention
from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_NUM_PROC


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
    ppl_results = []
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
        
        prepare_dataset = partial(prepare_dataset_fct,
                                  tokenizer=tokenizer,
                                  feature_extractor=feature_extractor)
        dataset = dataset.map(prepare_dataset, num_proc=DEFAULT_NUM_PROC)

        # Load data collator:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                             feature_extractor=feature_extractor,
                                                             return_attention_mask=True)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        # Placeholders for per-batch perplexities:
        ppl_per_batch: List[torch.Tensor] = []
        
        for batch in dataloader:
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = batch["input_features"].to(device).to(torch_dtype)
            tokenized_seq = concat_special_tokens(batch[DEFAULT_LABEL_TOKENIZED_COL].to(device),
                                                  pretrained_model_name_or_path,
                                                  language=language,
                                                  task=task)
            attention_mask = batch["attention_mask"].to(device)
            attention_mask_prefix = torch.Tensor([1, 1, 1, 1]).expand(attention_mask.shape[0], -1).to(attention_mask.device)
            attention_mask = torch.cat([attention_mask_prefix, attention_mask[:, 2:]], dim=1)

            # Shift inputs for next-word prediction:
            decoder_input_ids = tokenized_seq[:, 1:]  # [w1, w2, ..., wN, EOT]
            decoder_input_ids_right_shifted = tokenized_seq[:, :-1]  # [SOT, w1, w2, ..., wN]
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
            perplexity = torch.exp(-mean_log_prob_seq)  # (batch_size,)
            
            # Add to the list of perplexities:
            ppl_per_batch.append(perplexity)
        
        # Add to the list of perplexities:
        ppl_current_dataset = torch.cat(ppl_per_batch, dim=0).mean().item()
        ppl_results.append(ppl_current_dataset)
    
    # Save the results:
    results = pd.Series(ppl_results, index=list(ds_group.keys()), name="Perplexity")
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
