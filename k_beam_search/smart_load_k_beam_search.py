import os
from functools import partial
from typing import Any, Dict
from pathlib import Path

import torch

from transformers.models.whisper import WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer
from datasets import load_from_disk, DatasetDict

from dataloader.utils import remove_unnecessary_features_for_1_best
from k_beam_search.prepare_k_beam_features import prepare_k_beam_features_fct
from k_beam_search.ts_alignment_heads import MODEL_NAME_TO_ALIGNMENT_HEADS
from utils.distil_config import DistilConfig
from utils.file_io import extract_exp_name_from_model_path
from utils.process_tokenized_seq import remove_padding_fct
from utils.constants import DEFAULT_NUM_PROC, GEN_MAX_LENGTH



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
                                          dataset_dict: DatasetDict,
                                          teacher_caching_batch_size: int = 32) -> DatasetDict:
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
            print("No previously cached K-Beam search results could be found.")
        
        # Get the device:
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16  # see https://huggingface.co/learn/audio-course/chapter5/evaluation?fw=pt
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = torch.device('mps')
            torch_dtype = torch.float32  # float16 not supported by MPS
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        # Initialize the model from pretrained checkpoint:
        print(f"Loading teacher model for K-Beam search from `{config.teacher_model_name_or_path}`...")
        teacher_model = WhisperForConditionalGeneration.from_pretrained(config.teacher_model_name_or_path, torch_dtype=torch_dtype).to(device)
        if config.teacher_original_name is not None:
            print(f"Set the alignment heads of the teacher model using the config from `{config.teacher_original_name}`...")
            teacher_model.generation_config.alignment_heads = MODEL_NAME_TO_ALIGNMENT_HEADS[config.teacher_original_name]
        else:
            print(f"Set the alignment heads of the teacher model using the config from `{config.teacher_model_name_or_path}`...")
            teacher_model.generation_config.alignment_heads = MODEL_NAME_TO_ALIGNMENT_HEADS[config.teacher_model_name_or_path]
        
        # NOTE: BetterTransformer is not compatible with `return_token_timestamps`
        # Speed up inference if possible:
        # if device == "cuda:0":
        #     teacher_model = BetterTransformer.transform(teacher_model)

        teacher_model.generate = partial(teacher_model.generate, language=config.lang_name, task=config.task,
                                         max_length=GEN_MAX_LENGTH, return_token_timestamps=True, use_cache=True)
        
        if config.distillation_num_beams == 1:
            dataset_dict = dataset_dict.with_format("pt")

            def predict_seq_with_ts(batch: Dict[str, Any]) -> Dict[str, Any]:
                outputs = teacher_model.generate(batch["input_features"].to(device).to(torch_dtype))
                return {"teacher_sequences": outputs.sequences,
                        "token_timestamps": outputs.token_timestamps}
            
            print("Generating 1-Best output...")
            dataset_dict = dataset_dict.map(predict_seq_with_ts, batched=True, batch_size=teacher_caching_batch_size)

            print("Removing padding created by `model.generate()`...")
            remove_padding = partial(remove_padding_fct, col_sequences="teacher_sequences", col_timestamps="token_timestamps")
            dataset_dict = dataset_dict.map(remove_padding, num_proc=DEFAULT_NUM_PROC)
        
        else:
            prepare_k_beam_features = partial(prepare_k_beam_features_fct,
                                              model=teacher_model,
                                              num_beams=config.distillation_num_beams)
            print("Generating K-Beam search output...")
            dataset_dict = dataset_dict.with_format("pt").map(prepare_k_beam_features,
                                                              batched=True,
                                                              batch_size=teacher_caching_batch_size)
        
        # Select features of interest:
        print("Removing unnecessary features from the dataset...")
        for split in dataset_dict:
            dataset_dict[split] = remove_unnecessary_features_for_1_best(dataset_dict[split])
        
        # if device == "cuda:0":
        #     teacher_model = BetterTransformer.reverse(teacher_model)
        
        # Set the path to save the K-Beam search results:
        cache_filepath = str(parent_cache_dir / f"k_{config.distillation_num_beams}")
        
        Path(cache_filepath).parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(cache_filepath)
        print(f"Dataset with K-Beam search saved to `{cache_filepath}`. It will be loaded from disk next time.")
    
    return dataset_dict
