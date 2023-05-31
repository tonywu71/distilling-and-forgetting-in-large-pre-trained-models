from typing import Dict

import os
import shutil
from pathlib import Path

from transformers.generation import BeamSearchEncoderDecoderOutput
from datasets import Dataset

from k_beam_search.k_beam_search import get_k_beam_search_output
from utils.distil_config import DistilConfig
from utils.file_io import extract_exp_name_from_model_path
from utils.constants import K_BEAM_SEARCH_DIRNAME
from utils.pickle_caching import load_pickle, save_as_pickle


def smart_load_k_beam_search(config: DistilConfig,
                             dataset: Dataset) -> Dict[str, BeamSearchEncoderDecoderOutput]:
    
    # Sanity checks:
    assert config.method in ["seq_level_1_best", "seq_level_k_best_uniform", "seq_level_k_best_ranked"], \
        f'Invalid method `{config.method}`. Must be one of `["seq_level_1_best", "seq_level_k_best_uniform", "seq_level_k_best_ranked"]`.'
    assert config.distillation_num_beams is not None, \
        "The `distillation_num_beams` must be set for sequence-level distillation."
    
    processed_datasets_dir = Path(os.environ["PREPROCESSED_DATASETS_DIR"])
    if not processed_datasets_dir.exists():
        print(f"Preprocessed datasets directory `{os.environ['PREPROCESSED_DATASETS_DIR']}` not found. A new directory will be created.")
        processed_datasets_dir.mkdir(parents=True)
    
    
    # Extract the conventional experiment name from the teacher model path:
    teacher_ref = extract_exp_name_from_model_path(config.teacher_model_name_or_path)
    
    # Get the path to the preprocessed dataset:
    dataset_dir = str(Path(os.environ["PREPROCESSED_DATASETS_DIR"]) / K_BEAM_SEARCH_DIRNAME / \
                      config.dataset_name / teacher_ref / f"k_{config.distillation_num_beams}")
    # Example of generated `dataset_dir` -> "preprocessed_datasets/k_beam_search/ami_10h/openai/whisper-medium/k_5/"
    
    
    # Load the preprocessed dataset if it exists and if `force_reprocess_k_best` is set to `False`:
    if not config.force_reprocess_k_best and Path(dataset_dir).exists():
        print(f"Previously saved K-Beam search results found at `{dataset_dir}`. Loading from disk...")
        id_to_k_beam_search_output = load_pickle(dataset_dir)
    
    
    # Otherwise, preprocess the dataset from scratch:
    else:
        if config.force_reprocess_k_best:
            print(f"`force_reprocess_k_best` was set to `True` in the config file. ")
            if Path(dataset_dir).exists():
                print(f"Deleting previously saved K-Beam search results at `{dataset_dir}`...")
                shutil.rmtree(dataset_dir) 
        else:
            print(f"Previously saved K-Beam search results not found at `{dataset_dir}`.")
        
        
        print(f"Computing K-beam search from scratch...")
        id_to_k_beam_search_output = get_k_beam_search_output(model_name_or_path=config.teacher_model_name_or_path,
                                                              dataset=dataset,
                                                              col_id=config.col_id,
                                                              num_beams=config.distillation_num_beams)
        
        
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        save_as_pickle(id_to_k_beam_search_output, savepath=dataset_dir)
        print(f"K-Beam search results saved to `{dataset_dir}`. They will be loaded from disk next time.")
    
    return id_to_k_beam_search_output
