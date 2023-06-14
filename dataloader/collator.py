from typing import List, Dict, Tuple, Union

import torch
from transformers import WhisperProcessor

from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, LOSS_MASK_IDX


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Class to collate data for speech seq2seq models with padding.
    """
    
    def __init__(self,
                 processor: WhisperProcessor,
                 return_attention_mask: bool = False,
                 replace_padded_with_loss_mask_for_labels: bool = False,
                 add_k_beam_features: bool = False):
        self.processor = processor
        self.return_attention_mask = return_attention_mask
        self.replace_padded_with_loss_mask_for_labels = replace_padded_with_loss_mask_for_labels
        self.add_k_beam_features = add_k_beam_features
    
    
    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Split inputs and labels since they have to be of different lengths and need different padding methods.
        
        We expect `features` to be as such:
        >> [
        >>     {
        >>         "input_features": [...],
        >>         "labels": [...],
        >>         "input_ids": [...]
        >>     },
        >>     ...
        >> ]
        
        The default `features` are:
        >> {
        >>     "input_features": [...],
        >>     DEFAULT_LABEL_TOKENIZED_COL: [...]
        >> }
        """
        
        
        # --- Input features ---
        # `features`: list of `batch_size` dicts (each key is a column name)
        input_features = [{"input_features": feature["input_features"]} for feature in features]  # get only the feature of interest
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")  # type: ignore
        
        
        # --- Labels (tokenized) ---
        label_features = [{"input_ids": feature[DEFAULT_LABEL_TOKENIZED_COL]} for feature in features]  # get only the feature of interest
        labels, attention_mask_labels = self.preprocess_tokenized_labels(label_features,
                                                                         replace_padded_with_loss_mask=self.replace_padded_with_loss_mask_for_labels,
                                                                         discard_first_bos_token=True)
        batch[DEFAULT_LABEL_TOKENIZED_COL] = labels  # (batch_size, n_tokens)
        
        if self.return_attention_mask:
            batch["attention_mask_labels"] = attention_mask_labels  # (batch_size, n_tokens)
        
        
        # Add K-beam features if distillation:
        if self.add_k_beam_features:
            # --- Teacher sequences ---
            # Note: `tokenizer.pad` only accepts 1D-tensors which is not the case here as here they have shape (num_beams, n_tokens).
            #       However, we can take advantage of the fact that batch and beam dimensions are indifferent. Hence, we can simply
            #       iterate over the batch dimension and pad each tensor individually.
            batch_size = len(features)
            
            # Note: `teacher_sequences_features` contains the teacher sequences of all beams for all samples in the batch.
            #       This has nothing to do with the input features.
            
            teacher_sequences_features = []
            for feature in features:
                for row in feature["teacher_sequences"]:
                    teacher_sequences_features.append({"input_ids": row})  # get only the feature of interest
            
            # Important: We should not use the loss mask here as `teacher_sequences_features` will only be used as the reference sequence and
            #            thus cannot contain the special token `LOSS_MASK_IDX`.
            teacher_sequences, attention_mask_teacher_sequences = self.preprocess_tokenized_labels(teacher_sequences_features,
                                                                                                   replace_padded_with_loss_mask=False,
                                                                                                   discard_first_bos_token=False)  # (batch_size * num_beams, n_tokens)
            
            batch["teacher_sequences"] = teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            batch["attention_mask_teacher_sequences"] = attention_mask_teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            
            # --- Teacher sequences scores ---
            # No need to pad the scores as they are already of the same shape:
            batch["teacher_sequences_scores"] = torch.stack([feature["teacher_sequences_scores"] for feature in features], dim=0)  # (batch_size, num_beams)
        
        return batch
    
    
    def preprocess_tokenized_labels(self,
                                    features: List[Dict[str, Union[List[int], torch.Tensor]]],
                                    replace_padded_with_loss_mask: bool = True,
                                    discard_first_bos_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize, pad, and replace padding with correct token for correct loss computation.
        
        Note: Because `PADDING_IDX` token is used for padding and not for the vocabulary, we can simply use
              PADDING_IDX as a way to reconstruct the attention mask during loss computation.
        """
        
        # Pad the features:
        labels_batch = self.processor.tokenizer.pad(features, return_tensors="pt")  # type: ignore
        
        # Note: The output `labels_batch` contains the following keys: ["input_ids", "attention_mask"].
        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        
        if replace_padded_with_loss_mask:
            # Replace padding with correct token for correct loss computation:
            labels = labels.masked_fill(attention_mask.ne(1), LOSS_MASK_IDX)
        
        if discard_first_bos_token:
            # If a BOS ("Beginning Of Sequence") token was appended in previous tokenization step,
            # discard it as it will get appended later anyway:
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():  # type: ignore
                labels = labels[:, 1:]
        
        return labels, attention_mask
