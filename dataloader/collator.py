from typing import List, Dict, Tuple, Union

import torch
from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast, WhisperFeatureExtractor

from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, LOSS_MASK_IDX


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Class to collate data for speech seq2seq models with padding.
    """
    
    def __init__(self,
                 tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                 feature_extractor: WhisperFeatureExtractor,
                 return_attention_mask: bool = False,
                 replace_padded_with_loss_mask_for_labels: bool = False,
                 discard_first_bos_token: bool = False,
                 add_k_beam_features: bool = False):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.return_attention_mask = return_attention_mask
        self.replace_padded_with_loss_mask_for_labels = replace_padded_with_loss_mask_for_labels
        self.discard_first_bos_token = discard_first_bos_token
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
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        
        
        # --- Labels (tokenized) ---
        label_features = [{"input_ids": feature[DEFAULT_LABEL_TOKENIZED_COL]} for feature in features]  # get only the feature of interest
        labels, attention_mask = self.preprocess_tokenized_labels(label_features,
                                                                  replace_padded_with_loss_mask=self.replace_padded_with_loss_mask_for_labels,
                                                                  discard_first_bos_token=self.discard_first_bos_token)  # (batch_size, n_tokens)
        batch[DEFAULT_LABEL_TOKENIZED_COL] = labels  # (batch_size, n_tokens)
        
        if self.return_attention_mask:
            batch["attention_mask"] = attention_mask  # (batch_size, n_tokens)
        
        
        # Add K-beam features if distillation:
        if self.add_k_beam_features:
            # ==================== Teacher sequences ====================
            # NOTE: `tokenizer.pad` only accepts 1D-tensors which is not the case here as here they have shape (num_beams, n_tokens).
            #       However, we can take advantage of the fact that batch and beam dimensions are indifferent. Hence, we can simply
            #       iterate over the batch dimension and pad each tensor individually.
            batch_size = len(features)
            
            # NOTE: `teacher_sequences_features` contains the teacher sequences of all beams for all samples in the batch.
            #       This has nothing to do with the input features.
            
            teacher_sequences_features = []
            for feature in features:
                for row in feature["teacher_sequences"]:
                    teacher_sequences_features.append({"input_ids": row})  # get only the feature of interest
            
            # Important: We should not use the loss mask here as `teacher_sequences_features` will only be used as the reference sequence and
            #            thus cannot contain the special token `LOSS_MASK_IDX`.
            teacher_sequences, attention_mask_teacher_sequences = self.preprocess_tokenized_labels(teacher_sequences_features,
                                                                                                   replace_padded_with_loss_mask=self.replace_padded_with_loss_mask_for_labels,
                                                                                                   discard_first_bos_token=self.discard_first_bos_token)  # (batch_size * num_beams, n_tokens)
            
            batch["teacher_sequences"] = teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            batch["attention_mask_teacher_sequences"] = attention_mask_teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            
            # ==================== Teacher sequences scores ====================
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
        labels_batch = self.tokenizer.pad(features, return_tensors="pt")  # type: ignore
        
        # NOTE: With a fast tokenizer, using the `__call__` method is faster than using a method to encode the text
        #       followed by a call to the `pad` method to get a padded encoding. However, we have already tokenized
        #       the labels during preprocessing, so we can simply use the `pad` method here.
        
        # Get the labels and attention mask:
        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        
        if replace_padded_with_loss_mask:
            # Replace padding with correct token for correct loss computation:
            labels = labels.masked_fill(attention_mask.ne(1), LOSS_MASK_IDX)
        
        if discard_first_bos_token:
            # If a BOS ("Beginning Of Sequence") token was appended in previous tokenization step (which is
            # the case with the default Whisper tokenizer), discard it as it will get appended later anyway
            # when computing loss (see the `shift_tokens_right` method).
            if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():  # type: ignore
                labels = labels[:, 1:]
                attention_mask = attention_mask[:, 1:]
        
        return labels, attention_mask
