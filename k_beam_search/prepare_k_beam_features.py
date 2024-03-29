from typing import Dict, Any

import torch
from transformers.models.whisper import WhisperForConditionalGeneration

from utils.process_tokenized_seq import remove_padding_k_beam


def prepare_k_beam_features_fct(batch: Dict[str, Any],
                                model: WhisperForConditionalGeneration,
                                num_beams: int) -> Dict[str, Any]:
    """
    Utility to create K-Beam features for a dataset. Should be used with `Dataset.map()`.
    Note: `num_beams` must be > 1.
    
    Important: The dataset must be converted to PyTorch format first.
    
    The following new columns are added to the dataset:
    - `sequences` -> (num_beams, n_tokens)
    - `sequences_scores` -> (num_beams,)
    """
    
    assert num_beams > 1, f"Invalid `num_beams` value: {num_beams}. Must be > 1."
    
    device = model.device
    torch_dtype = model.dtype
    input_features = batch["input_features"].to(device).to(torch_dtype)
    
    # Generate teacher predictions using K-beam search:
    # NOTE: The outputs of `generate` are not affected by the precision dtype of the model.
    outputs = model.generate(input_features,
                             num_beams=num_beams,
                             num_return_sequences=num_beams,
                             output_scores=True,
                             return_dict_in_generate=True)
    
    # NOTE:
    # - outputs.sequences -> (batch_size * num_beams, n_tokens)
    # - outputs.sequences_scores -> (batch_size * num_beams,)
    
    # # Add the following fields to the current batch, i.e. a fortiori add columns for the dataset:
    batch["teacher_sequences"] = [
        remove_padding_k_beam(x) for x in torch.split(outputs.sequences, split_size_or_sections=num_beams, dim=0)
    ]  # `batch_size` tensors of shape (num_beams, n_tokens), unnecessary padding removed
    # batch["teacher_sequences"] = list(torch.split(outputs.sequences, split_size_or_sections=num_beams, dim=0))

    batch["teacher_sequences_scores"] = list(torch.split(outputs.sequences_scores,
                                                         split_size_or_sections=num_beams,
                                                         dim=0))  # `batch_size` tensors of shape (num_beams,)
    
    return batch
