import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

import pandas as pd
from tqdm.auto import tqdm

from transformers import (pipeline,
                          WhisperProcessor,
                          WhisperForConditionalGeneration)

from dataloader.dataloader import gen_from_dataset
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from evaluation.string_edit_metrics import get_string_edit_metrics
from normalization.whisper_normalization import get_whisper_normalizer


def eval_whisper_on_dataset_group(pretrained_model_name_or_path: str,
                                  ds_group: BaseDatasetGroup,
                                  task: str = "transcribe",
                                  num_beams: int = 1,
                                  batch_size: int = 16) -> pd.DataFrame:
    
    assert ds_group.is_preprocessed, "The dataset group must be preprocessed."
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."
    
    # Load model:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    # Loop over the datasets:
    dict_string_edit_metrics = defaultdict(list)
    
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        # Handle the special case of the English dataset with the basic normalizer:
        if language == "english-basic_normalizer":
            whisper_norm = get_whisper_normalizer(language=None)
            language = "english"
        else:
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
        
        for out in whisper_asr(gen_from_dataset(dataset),
                               batch_size=batch_size,
                               generate_kwargs={"num_beams": num_beams}):
            if not out["reference"][0].strip():  # type: ignore
                continue  # skip empty references to avoid error in WER computation
            predictions.append(whisper_norm(out["text"]))  # type: ignore
            references.append(out["reference"][0])  # type: ignore
        
        # Compute the WER in percent:
        string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=references, predictions=predictions))
        
        dict_string_edit_metrics["WER (%)"].append(string_edit_metrics["wer"])
        dict_string_edit_metrics["Sub (%)"].append(string_edit_metrics["sub"])
        dict_string_edit_metrics["Del (%)"].append(string_edit_metrics["del"])
        dict_string_edit_metrics["Ins (%)"].append(string_edit_metrics["ins"])
    
    # Create a DataFrame with the results:
    df_edit_metrics = pd.DataFrame(dict_string_edit_metrics, index=list(ds_group.keys()))
    df_edit_metrics.index.name = "Dataset"
    
    return df_edit_metrics
