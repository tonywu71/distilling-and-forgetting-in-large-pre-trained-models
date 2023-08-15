from typing import Dict, Any

import torch
from transformers.models.whisper import WhisperTokenizer, WhisperFeatureExtractor

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from trainer.trainer_utils import get_padded_mask_from_tensor

from utils.constants import LOSS_MASK_IDX


class DataCollatorWithPaddingForSeqLevelDistillation(DataCollatorSpeechSeq2SeqWithPadding):
    """
    Class to collate data for speech seq2seq models with padding during sequence distillation.

    NOTE: If `return_attention_mask` is True, then the attention mask will be computed from the labels.
          Elements with value 0 correspond to padding tokens. Thus, they should be masked out during training.
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
            label_features = [{"input_ids": feature["teacher_sequences"]} for feature in features]
            
            labels, attention_mask = self.preprocess_tokenized_labels(label_features,
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

            attention_mask_teacher_sequences = get_padded_mask_from_tensor(teacher_sequences)  # elements with value 0 correspond to padding tokens
            teacher_sequences = teacher_sequences.masked_fill(attention_mask_teacher_sequences.eq(0), LOSS_MASK_IDX)
            
            batch["teacher_sequences"] = teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            batch["attention_mask_teacher_sequences"] = attention_mask_teacher_sequences.reshape(batch_size, -1, teacher_sequences.shape[-1])  # (batch_size, num_beams, n_tokens)
            
            
            # ==================== Teacher sequences scores ====================
            # No need to pad the scores as they are already of the same shape:
            batch["teacher_sequences_scores"] = torch.stack([feature["teacher_sequences_scores"] for feature in features], dim=0)  # (batch_size, num_beams)
        
        return batch
