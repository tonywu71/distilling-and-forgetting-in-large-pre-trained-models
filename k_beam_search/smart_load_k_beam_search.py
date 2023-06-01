from typing import Dict

import os
import re
from pathlib import Path

from transformers import WhisperProcessor
from transformers.generation import BeamSearchEncoderDecoderOutput
from datasets import Dataset
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding

from k_beam_search.k_beam_search import get_k_beam_search_output
from utils.distil_config import DistilConfig
from utils.file_io import extract_exp_name_from_model_path
from utils.constants import DATASET_NAME_TO_COL_ID
from utils.pickle_caching import load_pickle, save_as_pickle


def smart_load_k_beam_search(config: DistilConfig,
                             dataset: Dataset) -> Dict[str, BeamSearchEncoderDecoderOutput]:
    
    # Sanity checks:
    assert config.method in ["seq_level_k_best_uniform", "seq_level_k_best_ranked"], \
        f'Invalid method `{config.method}`. Must be one of `["seq_level_k_best_uniform", "seq_level_k_best_ranked"]`.'
    assert config.distillation_num_beams is not None, \
        "The `distillation_num_beams` must be set for sequence-level distillation."
    assert config.dataset_name in DATASET_NAME_TO_COL_ID, \
        f"Invalid dataset name `{config.dataset_name}`. Must be one of `{list(DATASET_NAME_TO_COL_ID.keys())}`."
    
    
    processed_datasets_dir = Path(os.environ["K_BEAM_SEARCH_CACHE_DIR"])
    if not processed_datasets_dir.exists():
        print(f"Preprocessed datasets directory `{os.environ['K_BEAM_SEARCH_CACHE_DIR']}` not found. A new directory will be created.")
        processed_datasets_dir.mkdir(parents=True)
    
    
    # Extract the conventional experiment name from the teacher model path:
    teacher_ref = extract_exp_name_from_model_path(config.teacher_model_name_or_path)
    
    # Get the path to the preprocessed dataset:
    cache_dir = Path(os.environ["K_BEAM_SEARCH_CACHE_DIR"]) / config.dataset_name / teacher_ref
    
    # Create the directory if it does not exist:
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    list_suitable_cached_k = []
    
    # Get list of all Pickle files in `dataset_dir_all`:
    for x in cache_dir.iterdir():
        if x.is_file() and x.suffix == ".pkl":
            # Use Regex to retrieve the number of beams from the file name:
            # Example: "preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5.pkl" -> "5"
            reg_pattern = r'^k_\d+$'
            if re.match(reg_pattern, x.stem):
                cached_k = int(re.findall(reg_pattern, x.stem)[0])
                if cached_k >= config.distillation_num_beams:
                    list_suitable_cached_k.append(cached_k)
    
    # Use the cached K-Beam search results with the smallest K
    # that is >= `config.distillation_num_beams` for efficiency:
    cache_filepath = str(cache_dir / f"k_{min(list_suitable_cached_k)}.pkl") if list_suitable_cached_k else None
    
    # Example of generated `dataset_dir`:
    # - "preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5/" if there is a suitable cache
    # - None otherwise
    
    
    # Load the preprocessed dataset if it exists and if `force_reprocess_k_best` is set to `False`:
    if not config.force_reprocess_k_best and cache_filepath is not None:
        print(f"Previously saved K-Beam search results found at `{cache_filepath}`. Loading from disk...")
        id_to_k_beam_search_output = load_pickle(cache_filepath)
    
    
    # Otherwise, preprocess the dataset from scratch:
    else:
        if config.force_reprocess_k_best:
            print(f"`force_reprocess_k_best` was set to `True` in the config file. ")
            if cache_filepath is not None:
                print(f"Deleting previously saved K-Beam search results at `{cache_filepath}`...")
                os.remove(cache_filepath)
        else:
            print(f"Previously saved and suitable K-Beam search results could not be found in `{cache_dir}`.")
        
        
        # Load the data collator:
        processor = WhisperProcessor.from_pretrained(
            config.teacher_model_name_or_path,
            language=config.lang_name,
            task=config.task
        )
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
        
        print(f"\nComputing K-beam search from scratch...")
        id_to_k_beam_search_output = get_k_beam_search_output(model_name_or_path=config.teacher_model_name_or_path,
                                                              dataset=dataset,
                                                              col_id=DATASET_NAME_TO_COL_ID[config.dataset_name],
                                                              num_beams=config.distillation_num_beams,
                                                              data_collator=data_collator)
        
        # If we need to cache the K-Beam search results from scratch, K must be >= 2.
        # This is not a problem as we will simply use K=1 (≤ 2) for the distillation.
        k_beams = min(config.distillation_num_beams, 2)

        # Set the path to save the K-Beam search results:
        cache_filepath = str(cache_dir / f"k_{k_beams}.pkl")
        
        Path(cache_filepath).parent.mkdir(parents=True, exist_ok=True)
        save_as_pickle(data=id_to_k_beam_search_output, savepath=cache_filepath)
        print(f"K-Beam search results saved to `{cache_filepath}`. They will be loaded from disk next time.")
    
    return id_to_k_beam_search_output
