from typing import List, Dict, Union

import torch
from transformers import WhisperProcessor

from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, LOSS_MASK_IDX


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Class to collate data for speech seq2seq models with padding.
    """
    
    def __init__(self,
                 processor: WhisperProcessor,
                 is_distillation: bool=False):
        self.processor = processor
        self.is_distillation = is_distillation
    
    
    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Split inputs and labels since they have to be of different lengths and need different padding methods.
        
        We expect `features` to be as such:
        [
            {
                "input_features": [...],
                "labels": [...],
                "input_ids": [...]
            },
            
            ...
        ]
        
        The DataCollator will then return a batch of the following form:
        {
            "input_features": [...],
            DEFAULT_LABEL_TOKENIZED_COL: [...]
        }
        """
        
        # --- Input features ---
        # Get the input features and apply padding:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")  # type: ignore
        
        
        # --- Labels (tokenized) ---
        labels = self.preprocess_tokenized_labels(features, col_name=DEFAULT_LABEL_TOKENIZED_COL)
        batch[DEFAULT_LABEL_TOKENIZED_COL] = labels
        
        
        # Add K-beam features if distillation:
        if self.is_distillation:
            # --- Teacher sequences ---
            labels = self.preprocess_tokenized_labels(features, col_name="teacher_sequences")
            batch["teacher_sequences"] = labels
            
            # --- Teacher sequences scores ---
            # No need to pad the scores as they are already of the same shape:
            batch["teacher_sequences_scores"] = torch.stack([torch.tensor(feature["teacher_sequences_scores"], dim=0) for feature in features])
        
        return batch
    
    
    def preprocess_tokenized_labels(self,
                                    features: List[Dict[str, Union[List[int], torch.Tensor]]],
                                    col_name: str):
        """
        Tokenize, pad, and replace padding with correct token for correct loss computation.
        
        Note: Because `PADDING_IDX` token is used for padding and not for the vocabulary, we can simply use
              PADDING_IDX as a way to reconstruct the attention mask during loss computation.
        """
        
        # Get the tokenized label sequences and apply padding:
        label_features = [{"input_ids": feature[col_name]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")  # type: ignore
        # Note: `labels_batch` contains the following keys: ["input_ids", "attention_mask"]
        
        # Replace padding with correct token for correct loss computation:
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), LOSS_MASK_IDX)
        
        # If a BOS ("Beginning Of Sequence") token was appended in previous tokenization step,
        # discard it as it will get appended later anyway:
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():  # type: ignore
            labels = labels[:, 1:]
        
        return labels
