from typing import Dict, Tuple, Any

import torch
from transformers.models.whisper import WhisperTokenizer, WhisperFeatureExtractor

from utils.constants import DEFAULT_LABEL_TOKENIZED_COL, LOSS_MASK_IDX


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Class to collate data for speech seq2seq models with padding.
    """
    
    def __init__(self,
                 tokenizer: WhisperTokenizer,
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
        
        self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    
    def __call__(self,
                 features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
        
        return batch
    
    
    def preprocess_tokenized_labels(self,
                                    features: Dict[str, Any],
                                    replace_padded_with_loss_mask: bool = True,
                                    discard_first_bos_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize, pad, and replace padding with correct token for correct loss computation. Features are expected to
        be a list of dicts with a key `input_ids` containing the TOKENIZED labels.
        """
        
        # Pad the features:
        labels_batch = self.tokenizer.pad(features, return_tensors="pt")
        
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
            if (labels[:, 0] == self.sot_token).all().cpu().item():
                labels = labels[:, 1:]
                attention_mask = attention_mask[:, 1:]
        
        return labels, attention_mask
