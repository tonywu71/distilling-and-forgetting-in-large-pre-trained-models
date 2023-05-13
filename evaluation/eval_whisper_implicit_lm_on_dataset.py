import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

import pandas as pd
from tqdm.auto import tqdm

from transformers import WhisperProcessor
from evaluate import logging

from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from models.whisper_zero_cross_attention import WhisperForConditionalGenerationZeroCrossAttention
from utils.constants import DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_whisper_implicit_lm_on_dataset(pretrained_model_name_or_path: str,
                                        ds_group: BaseDatasetGroup,
                                        batch_size: int,  # TODO: Add support for batch_size > 1
                                        task: str="transcribe") -> pd.Series:
    
    assert ds_group.is_preprocessed, "The dataset group must be preprocessed."
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    # Load model:
    model_zero_cross_attention = WhisperForConditionalGenerationZeroCrossAttention.from_pretrained(pretrained_model_name_or_path)
    
    
    # Loop over the datasets:
    perplexity_results = []
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                     language=language,
                                                     task=task)
        
        model_zero_cross_attention.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)  # type: ignore
        
        # TODO: Replace `pipeline` with a 
        # whisper_asr = pipeline(task="automatic-speech-recognition",
        #                        model=model,
        #                        tokenizer=processor.tokenizer,  # type: ignore
        #                        feature_extractor=processor.feature_extractor,  # type: ignore
        #                        device=0  # use 1st GPU for Whisper
        # )
        
        # for start_index in logging.tqdm(range(0, len(dataset), batch_size)):
        #     end_index = min(start_index + batch_size, len(dataset))
        #     encoded_batch = dataset[start_index:end_index]
        #     # Collate the data into batches:
        #     data = self.data_collator(encoded_batch)  # type: ignore
        
        for data in dataset:
            # Collate the data into batches of size 1:
            data = self.data_collator([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device)
            tokenized_seq = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Shift inputs for next-word prediction:
            decoder_input_ids = tokenized_seq[:, 1:]
            shifted_left_decoder_input_ids = tokenized_seq[:, :-1]

            # One-step generation:
            output = model_zero_cross_attention.forward(input_features=input_features,  # type: ignore
                                                        decoder_input_ids=decoder_input_ids)

            # Convert logits to log-probabilities:
            log_prob_all = torch.nn.functional.log_softmax(output.logits, dim=-1)  # type: ignore
            
            # Take probabilities for the ground-truth tokens:
            log_prob = log_prob_all.take_along_dim(shifted_left_decoder_input_ids[..., None], dim=-1)
            
            # Compute perplexity:
            perplexity = torch.exp(-log_prob.mean()).item()
            
            # Add to the list of perplexities:
            perplexity_results.append(perplexity)
    
    
    # Save the results:
    results = pd.Series(perplexity_results, index=list(ds_group.keys()), name="WER (%)")
    results.index.name = "Dataset"
    
    # Compute the average WER:
    results["Average"] = results.mean()
    
    # Round the results:
    results = results.round(2)
    
    return results
