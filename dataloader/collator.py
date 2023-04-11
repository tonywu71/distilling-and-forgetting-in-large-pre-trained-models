from typing import List, Dict, Union

import torch
from transformers import WhisperProcessor

from utils.constants import PADDING_IDX


class DataCollatorSpeechSeq2SeqWithPadding:
    """Class to collate data for speech seq2seq models with padding."""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Split inputs and labels since they have to be of different lengths and need different padding methods.
        """
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")  # type: ignore

        # Get the tokenized label sequences:
        label_features = [
            {"input_ids": self.processor.tokenizer.truncate_sequences(feature["labels"])[0]} for feature in features]  # type: ignore
        
        # Pad the labels to max length:
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")  # type: ignore

        # Replace padding with correct token for correct loss computation:
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), PADDING_IDX)

        # If a BOS ("Beginning Of Sequence") token was appended in previous tokenization step,
        # discard it as it will get appended later anyway:
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():  # type: ignore
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
