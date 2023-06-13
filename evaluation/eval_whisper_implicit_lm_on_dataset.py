import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
assert torch.cuda.is_available(), "This script requires a GPU."

import pandas as pd
from tqdm.auto import tqdm

from transformers import WhisperProcessor

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from models.whisper_zero_cross_attention import WhisperForConditionalGenerationZeroCrossAttention
from normalization.whisper_normalization import get_whisper_normalizer
from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_whisper_implicit_lm_on_dataset(pretrained_model_name_or_path: str,
                                        ds_group: BaseDatasetGroup,
                                        task: str="transcribe") -> pd.Series:
    
    assert ds_group.is_preprocessed, "The dataset group must be preprocessed."
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    # Load model:
    model_zero_cross_attention = WhisperForConditionalGenerationZeroCrossAttention.from_pretrained(pretrained_model_name_or_path).to(device)  # type: ignore
    
    # Loop over the datasets:
    perplexity_results = []
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        
        # Handle the special case of the English dataset with the basic normalizer.
        # Note: `whisper_norm` is actually unused for perplexity computation but we
        # keep it for consistency with `eval_whisper_on_dataset`.
        if language == "english-basic_normalizer":
            whisper_norm = get_whisper_normalizer(language=None)
            language = "english"
        else:
            whisper_norm = get_whisper_normalizer(language=language)
        
        processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                     language=language,
                                                     task=task)
        
        # Load data collator:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor,
                                                             replace_padded_with_loss_mask_for_labels=True)
        
        # Set the forced decoder ids:
        model_zero_cross_attention.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)  # type: ignore
        
        # Placeholders for per-example perplexities:
        perplexities_curr_dataset = []
        
        for data in dataset:
            data = {
                "input_features": processor.feature_extractor(data["audio"]["array"],  # type: ignore
                                                              sampling_rate=processor.feature_extractor.sampling_rate).input_features[0],  # drop batch dimension  # type: ignore
                DEFAULT_LABEL_TOKENIZED_COL: processor.tokenizer(data[DEFAULT_LABEL_STR_COL]).input_ids  # type: ignore
            }
            
            # Collate the data into batches of size 1:
            data = data_collator([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device)
            tokenized_seq = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Shift inputs for next-word prediction:
            decoder_input_ids = tokenized_seq[:, :-1]
            shifted_left_decoder_input_ids = tokenized_seq[:, 1:]

            # One-step generation:
            output = model_zero_cross_attention.forward(input_features=input_features,  # type: ignore
                                                        decoder_input_ids=decoder_input_ids)  # type: ignore

            # Convert logits to log-probabilities:
            log_prob_all = torch.nn.functional.log_softmax(output.logits, dim=-1)  # type: ignore
            
            # Take probabilities for the ground-truth tokens:
            log_prob = log_prob_all.take_along_dim(shifted_left_decoder_input_ids[..., None], dim=-1)
            
            # Compute perplexity:
            perplexity = torch.exp(-log_prob.mean()).item()
            
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
