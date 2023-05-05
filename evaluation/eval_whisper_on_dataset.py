import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from transformers import (pipeline,
                          WhisperTokenizer,
                          WhisperProcessor,
                          WhisperForConditionalGeneration)
import evaluate

from dataloader.dataloader import gen_from_dataset
from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from normalization.whisper_normalization import get_whisper_normalizer

from utils.constants import DEFAULT_LABEL_STR_COL


def eval_whisper_on_dataset(pretrained_model_name_or_path: str,
                            ds_group: BaseDatasetGroup,
                            batch_size: int,
                            language: Optional[str]=None,
                            task: str="transcribe") -> pd.Series:
    
    if ds_group.is_multilingual:
        assert language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    # Load model:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    
    # Get normalizer:
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                 language=language,
                                                 task=task)
    whisper_norm = get_whisper_normalizer(tokenizer)
    
    def normalize_fct(batch):
        batch[DEFAULT_LABEL_STR_COL] = whisper_norm(batch["text"])
        return batch
    
    
    # Preprocess the datasets:
    ds_group.preprocess_datasets(normalize_fct=normalize_fct)
    
    
    # Load metric:
    wer_metric = evaluate.load("wer")
    
    
    # Loop over the datasets:
    wer_results = []
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        if not ds_group.is_multilingual:
            processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                         language=language,
                                                         task=task)
        else:
            processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                         language=ds_group.ds_name_to_lang[dataset_name],
                                                         task=task)
        
        whisper_asr = pipeline(task="automatic-speech-recognition",
                               model=model,
                               tokenizer=processor.tokenizer,  # type: ignore
                               feature_extractor=processor.feature_extractor,  # type: ignore
                               device=0  # use 1st GPU for Whisper
        )
        
        # Create placeholders for the predictions and references:
        predictions = []
        references = []
        
        for out in whisper_asr(gen_from_dataset(dataset), batch_size=batch_size):  # type: ignore
            if not out["reference"][0].strip():  # type: ignore
                continue  # skip empty references to avoid error in WER computation
            predictions.append(whisper_norm(out["text"]))  # type: ignore
            references.append(out["reference"][0])  # type: ignore
        
        # Compute the WER in percent:
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, ndigits=3)  # type: ignore
        
        wer_results.append(wer)
    
    
    # Save the results:
    results = pd.Series(wer_results, index=list(ds_group.keys()), name="WER (%)")
    results.index.name = "Dataset"
    
    # Compute the average:
    results["Average"] = results.mean()
    
    return results
