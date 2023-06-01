from typing import Dict

from tqdm.auto import tqdm

import torch

from transformers import WhisperForConditionalGeneration
from transformers.generation import BeamSearchEncoderDecoderOutput
from datasets import Dataset

from utils.constants import GEN_MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_k_beam_search_output(model_name_or_path: str,
                             dataset: Dataset,
                             col_id: str,
                             num_beams: int) -> Dict[str, BeamSearchEncoderDecoderOutput]:
    
    # Sanity check:
    assert col_id in dataset.features, f"Column `{col_id}` not found in dataset."
    if num_beams == 1:
        raise NotImplementedError("K-Beam search with K=1 is not supported during caching.")
    
    # Initialize the model from pretrained checkpoint:
    print(f"Loading model for K-Beam search from `{model_name_or_path}`...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)  # type: ignore
    
    id_to_k_beam_search_output: Dict[str, BeamSearchEncoderDecoderOutput] = {}
    
    for idx, data in tqdm(enumerate(dataset)):
        # Collate the data into batches of size 1:
        data = self.data_collator([data])  # type: ignore
        
        # Note that we need to move the data to the device manually (which is not the case with Trainer):
        input_features = data["input_features"].to(device)
        
        # Generate teacher predictions using K-beam search:
        id_to_k_beam_search_output[col_id] = model.generate(input_features,  # type: ignore
                                                            max_length=GEN_MAX_LENGTH,
                                                            num_beams=num_beams,
                                                            num_return_sequences=num_beams,
                                                            output_scores=True,
                                                            return_dict_in_generate=True)
    
    return id_to_k_beam_search_output


def get_batched_k_beam_search_output_from_inputs(inputs,
                                                 col_id: str,
                                                 distillation_num_beams: int,
                                                 id_to_k_beam_search_output: Dict[str, BeamSearchEncoderDecoderOutput]) -> BeamSearchEncoderDecoderOutput:
    """
    This function is used to get the K-Beam search output for a batch of inputs using the pre-computed K-beam results
    in `id_to_k_beam_search_output`. The returned object should be strictly identical to the output of `generate`
    on the batched `inputs` tensor.
    """
    
    batch_size, n_tokens = inputs.shape  # n_tokens is such that 1 <= n_tokens <= GEN_MAX_LENGTH
    beam_search_size = id_to_k_beam_search_output[col_id].sequences.shape[0]
    
    # Sanity checks:
    assert col_id in inputs.features, f"Column `{col_id}` not found in inputs."
    assert distillation_num_beams <= beam_search_size, \
        f"Invalid `distillation_num_beams` value `{distillation_num_beams}`. Must be <= `{beam_search_size}`."
    
    
    # Initialize the output tensors:
    sequences = torch.zeros((batch_size * distillation_num_beams, n_tokens), dtype=torch.long, device=device)  # (batch_size * distillation_num_beams, n_tokens)
    sequences_scores = torch.zeros((batch_size * distillation_num_beams,), dtype=torch.float, device=device)  # (batch_size * distillation_num_beams,)
    
    # Loop over the batch:
    for idx, sample in enumerate(inputs):
        # Get the inputs for the current sample:
        sample_id = sample[col_id]  # TODO: str or int???
        
        # Get the K-beam search output for the current sample:
        k_beam_search_output = id_to_k_beam_search_output[sample_id]
        
        # Get the sequence and its score:
        sequence = k_beam_search_output.sequences  # (beam_search_size, n_tokens)
        sequence_scores = k_beam_search_output.sequences_scores  # (beam_search_size,)
        
        # Store the sequence and its score in their respective tensors:
        sequences[idx:idx+distillation_num_beams, :len(sequence)] = sequence[:distillation_num_beams, :]
        sequences_scores[idx:idx+distillation_num_beams] = sequence_scores[:distillation_num_beams]  # type: ignore
    
    return BeamSearchEncoderDecoderOutput(sequences=sequences, sequences_scores=sequences_scores)  # type: ignore
