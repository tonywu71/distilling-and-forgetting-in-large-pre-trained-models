import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

import pandas as pd
from tqdm.auto import tqdm

from transformers import (pipeline,
                          WhisperProcessor,
                          WhisperForConditionalGeneration)
import evaluate

from dataloader.dataloader import gen_from_dataset
from dataloader.datasets.base_dataset_group import BaseDatasetGroup
from normalization.whisper_normalization import get_whisper_normalizer



def eval_whisper_on_dataset(pretrained_model_name_or_path: str,
                            ds_group: BaseDatasetGroup,
                            batch_size: int,
                            task: str="transcribe") -> pd.Series:
    
    assert ds_group.is_preprocessed, "The dataset group must be preprocessed."
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    # Load model:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    
    # Load metric:
    wer_metric = evaluate.load("wer")
    
    
    # Loop over the datasets:
    wer_results = []
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        whisper_norm = get_whisper_normalizer(language=language)
        
        processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                     language=language,
                                                     task=task)
        
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)  # type: ignore
        
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
        wer = 100 * wer  # type: ignore
        
        wer_results.append(wer)
    
    
    # Save the results:
    results = pd.Series(wer_results, index=list(ds_group.keys()), name="WER (%)")
    results.index.name = "Dataset"
    
    # Compute the average WER:
    results["Average"] = results.mean()
    
    # Round the results:
    results = results.round(2)
    
    return results
