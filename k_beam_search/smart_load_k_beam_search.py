from functools import partial

import os
import re
from pathlib import Path

import torch

from transformers import WhisperForConditionalGeneration
from datasets import load_from_disk, DatasetDict, Dataset

from k_beam_search.prepare_k_beam_features import prepare_k_beam_features_fct
from utils.distil_config import DistilConfig
from utils.file_io import extract_exp_name_from_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_k_beam_cache_dir(config: DistilConfig, parent_cache_dir: Path) -> str | None:
    """
    Returns the path to a suitable cached K-Beam search dataset. Returns `None` if no suitable cache is found.
    `cache_dir` is the path to the directory containing the cached K-Beam search datasets.
    
    Example of generated outputs: 
    - `"preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5/"`
    - `"preprocessed_datasets/k_beam_search/librispeech_dummy/openai/whisper-tiny/k_2/"`
    - `None`
    """
    
    assert config.distillation_num_beams is not None, \
        "The `distillation_num_beams` must be set for sequence-level distillation."
    
    list_suitable_cached_k = []
    
    # Get list of all the Pickle files in `dataset_dir_all`:
    for x in parent_cache_dir.iterdir():
        if x.is_dir():
            # Use Regex to retrieve the number of beams from the file name:
            # Example: "preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5.pkl" -> "5"
            reg_pattern = r'^k_(\d+)$'
            if re.match(reg_pattern, x.stem):
                cached_k = int(re.findall(reg_pattern, x.stem)[0])
                if cached_k >= config.distillation_num_beams:
                    list_suitable_cached_k.append(cached_k)
    
    # Use the cached K-Beam search results with the smallest K that is >= `config.distillation_num_beams` for efficiency:
    if list_suitable_cached_k:
        cache_filepath = str(parent_cache_dir / f"k_{min(list_suitable_cached_k)}")
    else:
        cache_filepath = None
    
    return cache_filepath


def smart_load_dataset_with_k_beam_search(config: DistilConfig,
                                          dataset_dict: DatasetDict | Dataset,
                                          zero_shot: bool = False) -> DatasetDict | Dataset:
    """
    Return a dataset with K-Beam search results. If a suitable cached dataset is found, it is loaded from disk.
    Otherwise, the dataset is preprocessed from scratch.
    
    The following new columns are added to the dataset:
    - `sequences` -> (num_beams, n_tokens)
    - `sequences_scores` -> (num_beams,)
    """
    
    # Sanity checks:
    assert config.method_distil in ["seq_level_k_best_uniform", "seq_level_k_best_ranked"], \
        f'Invalid method `{config.method_distil}`. Must be one of `["seq_level_k_best_uniform", "seq_level_k_best_ranked"]`.'
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
        
        # If we need to cache the K-Beam search results from scratch, K must be >= 2.
        # This is not a problem as we will simply use K=1 (≤ 2) for the distillation.
        if config.distillation_num_beams == 1:
            print("K has been temporarily set to 2 for the K-beam caching for compatibility reasons " + \
                  "but K=1 will still be used for the distillation.")
            num_beams = 2
        else:
            num_beams = config.distillation_num_beams
        
        # Initialize the model from pretrained checkpoint:
        print(f"Loading teacher model for K-Beam search from `{config.teacher_model_name_or_path}`...")
        model = WhisperForConditionalGeneration.from_pretrained(config.teacher_model_name_or_path).to(device)  # type: ignore
        
        if zero_shot:
            model.config.forced_decoder_ids = []
        else:
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=config.lang_name, task=config.task)  # type: ignore
        
        # Get the mapping function:
        prepare_k_beam_features = partial(prepare_k_beam_features_fct,
                                          model=model,
                                          num_beams=num_beams)
        
        # Map the dataset:
        print("\nGenerating K-Beam search output...")
        dataset_dict = dataset_dict.with_format("pt").map(prepare_k_beam_features,
                                                          batched=True,
                                                          batch_size=config.batch_size)
        
        # Set the path to save the K-Beam search results:
        cache_filepath = str(parent_cache_dir / f"k_{num_beams}")
        
        Path(cache_filepath).parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(cache_filepath)
        print(f"Dataset with K-Beam search saved to `{cache_filepath}`. It will be loaded from disk next time.")
    
    return dataset_dict
