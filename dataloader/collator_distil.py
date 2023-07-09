from typing import Dict, Tuple, Any

import torch
from transformers.models.whisper import WhisperTokenizer, WhisperFeatureExtractor

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.utils import get_fast_tokenizer
from trainer.trainer_utils import get_language_special_token, get_padded_mask_from_tensor, get_task_special_token

from utils.constants import DEFAULT_TOKENIZER_MAX_LENGTH, LOSS_MASK_IDX


class DataCollatorWithPaddingForSeqLevelDistillation(DataCollatorSpeechSeq2SeqWithPadding):
    """
    Class to collate data for speech seq2seq models with padding during sequence distillation.
    """
    
    def __init__(self,
                 tokenizer: WhisperTokenizer,
                 feature_extractor: WhisperFeatureExtractor,
                 return_attention_mask: bool = False,
                 replace_padded_with_loss_mask_for_labels: bool = False,
                 discard_first_bos_token: bool = False,
                 distillation_k_beam: int = 1):
        super().__init__(tokenizer=tokenizer,
                         feature_extractor=feature_extractor,
                         return_attention_mask=return_attention_mask,
                         replace_padded_with_loss_mask_for_labels=replace_padded_with_loss_mask_for_labels,
                         discard_first_bos_token=discard_first_bos_token)
        self.distillation_k_beam = distillation_k_beam
    
    
    def __call__(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Split inputs and labels since they have to be of different lengths and need different padding methods.
        Handles the case of sequence-level distillation.
        """
        
        # Get the default collated batch:
        batch = super().__call__(features)
        
        if self.distillation_k_beam == 1:
            # --- Teacher labels (NON-tokenized) ---
            label_features = [feature["teacher_text"].lower() for feature in features]  # get only the feature of interest
            
            # NOTE: The text has not been lowercased yet. We have to do it here as Whisper has been trained on lowercased text.
            
            labels, attention_mask = self.preprocess_untokenized_labels(label_features,
                                                                        replace_padded_with_loss_mask=self.replace_padded_with_loss_mask_for_labels,
                                                                        discard_first_bos_token=self.discard_first_bos_token)  # (batch_size, n_tokens)
            batch["teacher_sequences"] = labels  # (batch_size, n_tokens)
            
            if self.return_attention_mask:
                batch["attention_mask_teacher_sequences"] = attention_mask  # (batch_size, n_tokens)
        
        else:
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
            
            # NOTE: Right now, `attention_mask_teacher_sequences` is not correct because it doesn't take into account the padding performed by the generate method during
            #       caching. Therefore, we will have to recompute the attention mask here.
            attention_mask_teacher_sequences = get_padded_mask_from_tensor(teacher_sequences)
            teacher_sequences = teacher_sequences.masked_fill(attention_mask_teacher_sequences.eq(1), LOSS_MASK_IDX)
            
            batch["teacher_sequences"] = teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            batch["attention_mask_teacher_sequences"] = attention_mask_teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            
            
            # ==================== Teacher sequences scores ====================
            # No need to pad the scores as they are already of the same shape:
            batch["teacher_sequences_scores"] = torch.stack([feature["teacher_sequences_scores"] for feature in features], dim=0)  # (batch_size, num_beams)
        
        return batch
    
    
    def preprocess_untokenized_labels(self,
                                      features: Dict[str, Any],
                                      replace_padded_with_loss_mask: bool = True,
                                      discard_first_bos_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize, pad, and replace padding with correct token for correct loss computation. Features are expected to
        be a list of dicts with a key `input_ids` containing the raw string labels (UNTOKENIZED).
        """
        
        # Get the fast tokenizer:
        fast_tokenizer = get_fast_tokenizer(self.tokenizer)
        
        # Pad the features:
        labels_batch = fast_tokenizer(features, padding=True, truncation=True, max_length=DEFAULT_TOKENIZER_MAX_LENGTH, return_tensors="pt")
        
        # IMPORTANT: There is a bug in the current version of transformers (4.30.2) that makes
        #            `WhisperTokenizerFast` not work properly. It would forget to output the special tokens
        #            for `language` and `task`.
        # HOTFIX: Concatenate the special tokens to the vocabulary of the fast tokenizer manually.
        assert self.tokenizer.language is not None, "The tokenizer must have a language set."
        language_token = get_language_special_token(self.tokenizer.language)
        assert self.tokenizer.task is not None, "The tokenizer must have a task set."
        task_token = get_task_special_token(self.tokenizer.task)
        missing_special_tokens = torch.LongTensor([language_token, task_token]).expand(labels_batch["input_ids"].shape[0], -1)
        labels_batch["input_ids"] = torch.cat([labels_batch["input_ids"][..., 0:1], missing_special_tokens, labels_batch["input_ids"][..., 1:]], axis=1)
        labels_batch["attention_mask"] = torch.cat([labels_batch["attention_mask"][..., 0:1], torch.ones_like(missing_special_tokens), labels_batch["attention_mask"][..., 1:]], axis=1)
        
        # Get the labels and attention mask:
        tokenized_labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        
        if replace_padded_with_loss_mask:
            # Replace padding with correct token for correct loss computation:
            tokenized_labels = tokenized_labels.masked_fill(attention_mask.ne(1), LOSS_MASK_IDX)
        
        if discard_first_bos_token:
            # If a BOS ("Beginning Of Sequence") token was appended in previous tokenization step (which is
            # the case with the default Whisper tokenizer), discard it as it will get appended later anyway
            # when computing loss (see the `shift_tokens_right` method).
            if (tokenized_labels[:, 0] == self.sot_token).all().cpu().item():
                tokenized_labels = tokenized_labels[:, 1:]
                attention_mask = attention_mask[:, 1:]
        
        return tokenized_labels, attention_mask
