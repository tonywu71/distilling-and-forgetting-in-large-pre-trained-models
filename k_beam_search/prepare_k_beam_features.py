from typing import Dict, Any

from tqdm.auto import tqdm

import torch

from transformers import WhisperForConditionalGeneration
from transformers.generation import BeamSearchEncoderDecoderOutput
from datasets import Dataset

from utils.constants import GEN_MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_k_beam_features_fct(batch: Dict[str, Any],
                                model: WhisperForConditionalGeneration,
                                num_beams: int) -> Dict[str, Any]:
    """
    Utility to create K-Beam features for a dataset. Should be used with `Dataset.map()`.
    
    The following new columns are added to the dataset:
    - `sequences` -> (num_beams, n_tokens)
    - `sequences_scores` -> (num_beams,)
    """
    
    input_features = batch["input_features"]

    # Generate teacher predictions using K-beam search:
    outputs = model.generate(input_features,
                             max_length=GEN_MAX_LENGTH,
                             num_beams=num_beams,
                             num_return_sequences=num_beams,
                             output_scores=True,
                             return_dict_in_generate=True)
    # Note:
    # - outputs.sequences -> (batch_size * num_beams, n_tokens)
    # - outputs.sequences_scores -> (batch_size * num_beams,)
    
    # Add the follwoing fields to the current batch, i.e. a fortiori add columns for the dataset:
    batch["sequences"] = list(torch.split(outputs.sequences,
                                          split_size_or_sections=num_beams,
                                          dim=0))  # batch_size tensors of shape (num_beams, n_tokens)
    batch["sequences_scores"] = list(torch.split(outputs.sequences_scores,
                                                 split_size_or_sections=num_beams,
                                                 dim=0))  # batch_size tensors of shape (num_beams,)
    
    return batch
