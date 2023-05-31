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
        id_to_k_beam_search_output[col_id] = model.generate(input_features,
                                                            max_length=GEN_MAX_LENGTH,
                                                            num_beams=num_beams,
                                                            num_return_sequences=num_beams,
                                                            output_scores=True,
                                                            return_dict_in_generate=True)
    
    return id_to_k_beam_search_output
