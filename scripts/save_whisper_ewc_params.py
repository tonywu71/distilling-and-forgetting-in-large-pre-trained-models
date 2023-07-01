import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from typing import Dict, Tuple
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader

from transformers import PreTrainedModel
from transformers.models.whisper import (WhisperTokenizerFast,
                                         WhisperFeatureExtractor,
                                         WhisperForConditionalGeneration)
from transformers.models.whisper import WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataloader import load_dataset_dict
from trainer.estimate_ewc import get_mean_params, get_fisher_params

from utils.constants import EWC_PARAMS_VANILLA


def get_ewc_params(model: PreTrainedModel,
                   dataloader: DataLoader) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns the EWC parameters of the model.
    """
    mean_params = get_mean_params(model)
    fisher_params = get_fisher_params(model, dataloader)
    return mean_params, fisher_params


def get_dirpath_ewc_params(pretrained_model_name_or_path: str) -> str:
    if pretrained_model_name_or_path.startswith("openai/whisper-"):
        EWC_PARAMS_VANILLA.mkdir(parents=True, exist_ok=True)        
        return str(EWC_PARAMS_VANILLA / pretrained_model_name_or_path.split("openai/whisper-")[-1])
    else:
        return pretrained_model_name_or_path


def get_ewc_params_for_whisper(pretrained_model_name_or_path: str,
                               language: str,
                               task: str,
                               dataset_name: str,
                               batch_size: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    
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
    if device == "cuda:0":
        model = BetterTransformer.transform(model)
    
    # Initialize the tokenizer and feature extractor:
    tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path, language=language, task=task)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    
    # Get the dataloader:
    ds = load_dataset_dict(dataset_name)["train"]
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                         feature_extractor=feature_extractor,
                                                         return_attention_mask=True,
                                                         replace_padded_with_loss_mask_for_labels=False,
                                                         discard_first_bos_token=True)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    # Get the EWC params:
    mean_params, fisher_params = get_ewc_params(model, dataloader)
    
    return mean_params, fisher_params


def main(pretrained_model_name_or_path: str = typer.Argument(..., help="The name or path of the pretrained model."),
         language: str = typer.Option(..., help="The language of the pretrained model."),
         task: str = typer.Option(..., help="The task of the pretrained model."),
         dataset_name: str = typer.Option(..., help="The name of the dataset."),
         batch_size: int = typer.Option(1, help="The batch size for the dataloader.")):
    
    # Get the EWC params:
    mean_params, fisher_params = get_ewc_params_for_whisper(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                            language=language,
                                                            task=task,
                                                            dataset_name=dataset_name,
                                                            batch_size=batch_size)
    
    # Dump the EWC params as pickle files:
    dirpath = get_dirpath_ewc_params(pretrained_model_name_or_path)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(dirpath, "mean_params.pkl"), "wb") as f:
        pickle.dump(mean_params, f)
        print(f"Dumped EWC mean params to `{os.path.join(dirpath, 'mean_params.pkl')}`.")
    
    with open(os.path.join(dirpath, "fisher_params.pkl"), "wb") as f:
        pickle.dump(fisher_params, f)
        print(f"Dumped EWC fisher params to `{os.path.join(dirpath, 'fisher_params.pkl')}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
