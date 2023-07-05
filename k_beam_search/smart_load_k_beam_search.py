from functools import partial

import os
from pathlib import Path

import torch

from transformers.models.whisper import WhisperTokenizerFast, WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer
from datasets import load_from_disk, DatasetDict, Dataset

from k_beam_search.prepare_k_beam_features import prepare_k_beam_features_fct
from utils.distil_config import DistilConfig
from utils.file_io import extract_exp_name_from_model_path



def get_k_beam_cache_dir(config: DistilConfig, parent_cache_dir: Path) -> str | None:
    """
    Returns the path to the cached K-Beam search dataset if exists. Otherwise, returns `None`.
    Note: `parent_cache_dir` is the path to the directory containing the cached K-Beam search datasets.
    
    Example of generated outputs: 
    - `"preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5/"`
    - `"preprocessed_datasets/k_beam_search/librispeech_dummy/openai/whisper-tiny/k_2/"`
    - `None`
    """
    
    assert config.distillation_num_beams is not None, \
        "The `distillation_num_beams` must be set for sequence-level distillation."
    
    for x in parent_cache_dir.iterdir():
        if x.is_dir() and x.stem == f"k_{config.distillation_num_beams}":
            return str(x)
    
    return None


def smart_load_dataset_with_k_beam_search(config: DistilConfig,
                                          dataset_dict: DatasetDict | Dataset,
                                          teacher_caching_batch_size: int = 32) -> DatasetDict | Dataset:
    """
    Return a dataset with K-Beam search results. If a suitable cached dataset is found, it is loaded from disk.
    Otherwise, the dataset is preprocessed from scratch.
    
    The following new columns are added to the dataset:
    - `sequences` -> (num_beams, n_tokens)
    - `sequences_scores` -> (num_beams,)
    """
    
    # Sanity checks:
    assert config.method_distil in ["seq_level_uniform", "seq_level_ranked"], \
        f'Invalid method `{config.method_distil}`. Must be one of `["seq_level_uniform", "seq_level_ranked"]`.'
    assert config.distillation_num_beams is not None, \
        "The `distillation_num_beams` must be set for sequence-level distillation."
    
    # Get the path to the preprocessed dataset:
    k_beam_cache_dir = Path(os.environ["K_BEAM_SEARCH_CACHE_DIR"])
    if not k_beam_cache_dir.exists():
        print(f"Preprocessed datasets directory `{os.environ['K_BEAM_SEARCH_CACHE_DIR']}` not found. A new directory will be created.")
        k_beam_cache_dir.mkdir(parents=True)
    
    # Extract the conventional experiment name from the teacher model path:
    teacher_ref = extract_exp_name_from_model_path(config.teacher_model_name_or_path)
    
    # Get the path to the preprocessed dataset:
    parent_cache_dir = Path(os.environ["K_BEAM_SEARCH_CACHE_DIR"]) / config.dataset_name / teacher_ref
    
    # Create the directory if it does not exist:
    parent_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the path to the cached K-Beam search results:
    k_beam_cache_dir = get_k_beam_cache_dir(config, parent_cache_dir)  # will be `None` if no suitable cache is found
    
    
    # Load the preprocessed dataset if it exists and if `force_reprocess_k_best` is set to `False`:
    if not config.force_reprocess_k_best and k_beam_cache_dir is not None:
        print(f"Previously saved K-Beam search results found at `{k_beam_cache_dir}`. Loading from disk...")
        dataset_dict = load_from_disk(k_beam_cache_dir)
        print(f"Succesfully loaded dataset with K-Beam search from `{k_beam_cache_dir}`.")

    # Otherwise, preprocess the dataset from scratch:
    else:
        if config.force_reprocess_k_best:
            print(f"`force_reprocess_k_best` was set to `True` in the config file. ")
            if k_beam_cache_dir is not None:
                print(f"Deleting previously saved K-Beam search results at `{k_beam_cache_dir}`...")
                os.remove(k_beam_cache_dir)
        else:
            print(f"Previously saved and suitable K-Beam search results could not be found in `{parent_cache_dir}`.")
        
        # Get the device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model from pretrained checkpoint:
        print(f"Loading teacher model for K-Beam search from `{config.teacher_model_name_or_path}`...")
        model = WhisperForConditionalGeneration.from_pretrained(config.teacher_model_name_or_path).to(device)
        
        # Speed up inference if possible:
        if device == "cuda:0":
            model = BetterTransformer.transform(model)
        model.generate = partial(model.generate, language=config.lang_name, task=config.task, use_cache=True)
        
        if config.distillation_num_beams == 1:
            dataset_dict = dataset_dict.with_format("pt")
            dataset_dict = dataset_dict.map(lambda batch: {"teacher_sequence": model.generate(batch["input_features"].to(device), max_length=255)},
                                            batched=True, batch_size=teacher_caching_batch_size)
            tokenizer = WhisperTokenizerFast.from_pretrained(config.teacher_model_name_or_path, 
                                                             language=config.lang_name,
                                                             task=config.task)
            dataset_dict = dataset_dict.map(lambda batch: {"teacher_labels": tokenizer.batch_decode(batch["teacher_sequence"], skip_special_tokens=True)},
                                            batched=True,  # use default batch size for decoding
                                            remove_columns=["teacher_sequence"])
        else:
            prepare_k_beam_features = partial(prepare_k_beam_features_fct,
                                              model=model,
                                              num_beams=config.distillation_num_beams)
            print("\nGenerating K-Beam search output...")
            dataset_dict = dataset_dict.with_format("pt").map(prepare_k_beam_features,
                                                              batched=True,
                                                              batch_size=teacher_caching_batch_size)
        
        # Set the path to save the K-Beam search results:
        cache_filepath = str(parent_cache_dir / f"k_{config.distillation_num_beams}")
        
        Path(cache_filepath).parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(cache_filepath)
        print(f"Dataset with K-Beam search saved to `{cache_filepath}`. It will be loaded from disk next time.")
    
    return dataset_dict
