from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import pandas as pd

import torch
from transformers.models.whisper import (WhisperTokenizer,
                                         WhisperTokenizerFast,
                                         WhisperFeatureExtractor,
                                         WhisperForConditionalGeneration)
from transformers.pipelines import pipeline
from optimum.bettertransformer import BetterTransformer

from dataloader.dataset_loader import gen_from_dataset
from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from evaluation.string_edit_metrics import get_string_edit_metrics
from evaluation.eval_whisper_utils import save_preds_to_json
from normalization.whisper_normalization import get_whisper_normalizer
from utils.file_io import extract_output_savepath_from_model_path

from utils.constants import DEFAULT_EVAL_BATCH_SIZE, DEFAULT_LABEL_STR_COL, GEN_MAX_LENGTH


def eval_whisper_wer_on_dataset_group(pretrained_model_name_or_path: str,
                                      ds_group: BaseDatasetGroup,
                                      task: str = "transcribe",
                                      zero_shot: bool = False,
                                      num_beams: int = 1,
                                      batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
                                      fast_tokenizer: bool = True,
                                      save_preds: bool = False,
                                      generate_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Evaluate a Whisper model on a dataset group and return a DataFrame with the results.
    """
    
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
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype).to(device)
    
    if device == "cuda:0":
        model = BetterTransformer.transform(model)
    
    # Loop over the datasets:
    dict_string_edit_metrics = defaultdict(list)
    tbar = tqdm(ds_group.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Evaluating {dataset_name}...")
        
        if not ds_group.is_multilingual:
            language = ds_group.language
        else:
            language = ds_group.ds_name_to_lang[dataset_name]
        
        # Get normalizer (depends on the language of the current dataset):
        whisper_norm = get_whisper_normalizer(language=language)
        
        if fast_tokenizer:
            tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        else:
            tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        
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
        generate_kwargs = generate_kwargs.copy() if generate_kwargs else {}
        generate_kwargs.update({"max_length": GEN_MAX_LENGTH, "num_beams": num_beams})
        if not zero_shot:
            generate_kwargs.update({"language": language, "task": task})
        
        num_rows = dataset.num_rows if hasattr(dataset, "num_rows") else None
        
        for out in tqdm(whisper_asr(gen_from_dataset(dataset),
                                    batch_size=batch_size,
                                    generate_kwargs=generate_kwargs),
                        total=num_rows):
            ref = out["reference"][0].lower()
            pred = out[DEFAULT_LABEL_STR_COL]
            references.append(ref)
            predictions.append(pred)
        
        if save_preds:
            print()
            savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}-preds_orthographic.json"
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            save_preds_to_json(references, predictions, savepath)
            print(f"Exported orthographic references and predictions to `{savepath}`.")
        
        # Compute the orthographic WER in percent and save it in the dictionary:
        string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=references, predictions=predictions))
        dict_string_edit_metrics["WER ortho (%)"].append(string_edit_metrics["wer"])
        dict_string_edit_metrics["Sub ortho (%)"].append(string_edit_metrics["sub"])
        dict_string_edit_metrics["Del ortho (%)"].append(string_edit_metrics["del"])
        dict_string_edit_metrics["Ins ortho (%)"].append(string_edit_metrics["ins"])
        
        # Get the normalized references and predictions (overwrites the previous lists to save memory):
        predictions = list(map(whisper_norm, predictions))
        references = list(map(whisper_norm, references))
        
        if save_preds:
            savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + f"-{dataset_name}-preds_normalized.json"
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            save_preds_to_json(references, predictions, savepath)
            print(f"Exported orthographic references and predictions to `{savepath}`.")
            print()
        
        # Compute the normalized WER in percent and save it in the dictionary:
        string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=references, predictions=predictions))
        dict_string_edit_metrics["WER (%)"].append(string_edit_metrics["wer"])
        dict_string_edit_metrics["Sub (%)"].append(string_edit_metrics["sub"])
        dict_string_edit_metrics["Del (%)"].append(string_edit_metrics["del"])
        dict_string_edit_metrics["Ins (%)"].append(string_edit_metrics["ins"])
    
    # Create a DataFrame with the results:
    df_edit_metrics = pd.DataFrame(dict_string_edit_metrics, index=list(ds_group.keys()))
    df_edit_metrics.index.name = "Dataset"
    
    return df_edit_metrics
