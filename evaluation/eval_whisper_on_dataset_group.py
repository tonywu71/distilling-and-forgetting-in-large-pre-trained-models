import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

import torch

import pandas as pd
from tqdm.auto import tqdm

from transformers import (pipeline,
                          WhisperTokenizer,
                          WhisperTokenizerFast,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration)

from dataloader.dataloader import gen_from_dataset
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from evaluation.string_edit_metrics import get_string_edit_metrics
from normalization.whisper_normalization import get_whisper_normalizer
from utils.constants import DEFAULT_LABEL_STR_COL


def eval_whisper_on_dataset_group(pretrained_model_name_or_path: str,
                                  ds_group: BaseDatasetGroup,
                                  task: str = "transcribe",
                                  zero_shot: bool = False,
                                  num_beams: int = 1,
                                  batch_size: int = 64,
                                  fast_tokenizer: bool = True) -> pd.DataFrame:
    """
    Evaluate a Whisper model on a dataset group and return a DataFrame with the results.
    """
    
    if ds_group.is_multilingual:
        assert ds_group.language is None, "Language must be `None` for multilingual datasets as it is inferred from the BaseDatasetGroup's metadata."

    try:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            torch_dtype = torch.float32  # float16 not supported by MPS
        else:
            device = "cpu"
            torch_dtype = torch.float32
    except:
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16  # see https://huggingface.co/learn/audio-course/chapter5/evaluation?fw=pt
        else:
            device = "cpu"
            torch_dtype = torch.float32
    
    # Load model:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
    
    # Loop over the datasets:
    dict_string_edit_metrics = defaultdict(list)
    
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Evaluating {dataset_name}...")
        
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
        
        if fast_tokenizer:
            tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        else:
            tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        # Note: There is no need to set `language` and `task` for the processor here as the special tokens will be removed
        #       from the input text before comparison.
        
        
        # Create pipeline:
        whisper_asr = pipeline(task="automatic-speech-recognition",
                               model=model,
                               tokenizer=tokenizer,
                               feature_extractor=feature_extractor,
                               torch_dtype=torch_dtype,
                               device=device)
    
        # Create placeholders for the predictions and references:
        predictions = []
        references = []
        
        # Prepare the generation kwargs:
        generate_kwargs = {"num_beams": num_beams}
        if not zero_shot:
            generate_kwargs.update({"language": language, "task": task})
        
        num_rows = dataset.num_rows if hasattr(dataset, "num_rows") else None
        
        for out in tqdm(whisper_asr(gen_from_dataset(dataset),
                                    batch_size=batch_size,
                                    generate_kwargs=generate_kwargs),
                        total=num_rows):
            ref = whisper_norm(out["reference"][0])
            pred = whisper_norm(out[DEFAULT_LABEL_STR_COL])
            
            if not ref.strip():
                continue  # skip empty references to avoid error in WER computation
            
            references.append(ref)
            predictions.append(pred)
        
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
