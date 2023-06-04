from typing import Optional, Literal

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (TrainingArguments,
                          Trainer,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          PreTrainedModel,
                          WhisperTokenizer)
from transformers.modeling_outputs import Seq2SeqLMOutput


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method: Literal["word_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"],
                 ce_alpha: float,
                 temperature: Optional[float] = None,
                 distillation_num_beams: Optional[int] = None,
                 decay_beta: Optional[float] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.ce_alpha = ce_alpha
        self.temperature = temperature
        self.distillation_num_beams = distillation_num_beams
        self.decay_beta = decay_beta


class DistillationTrainer(Seq2SeqTrainer):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 student_tokenizer: Optional[WhisperTokenizer] = None,
                 teacher_model: Optional[PreTrainedModel] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        
        # Sanity checks:
        if self.args.method == "word_level":
            assert self.teacher_model is not None, \
                "The `teacher_model` must be set for word-level distillation."
        
        
        self.METHOD_TO_LOSS_FCT = {
            "word_level": self._compute_loss_word_level,
            "seq_level_k_best_uniform": self._compute_loss_seq_level_k_best_uniform,
            "seq_level_k_best_ranked": self._compute_loss_seq_level_k_best_ranked
        }
    
    
    def compute_loss(self,
                     student_model: PreTrainedModel,
                     inputs,
                     return_outputs: bool = False):
        """
        Computes the loss according to the distillation method specified in `self.args.method`.
        """
        loss, output_student = self.METHOD_TO_LOSS_FCT[self.args.method](student_model, inputs)
        return (loss, output_student) if return_outputs else loss
    
    
    def _compute_loss_word_level(self,
                                 student_model: PreTrainedModel,
                                 inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for word-level distillation.
        """
        
        assert self.teacher_model is not None, \
            "The `teacher_model` must be set for word-level distillation."
        
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels']
        
        # Forward pass through student:
        output_student: Seq2SeqLMOutput = student_model(**inputs)
        
        # Extract cross-entropy loss and logits from student
        loss_ce = output_student.loss
        logits_student = output_student.logits
        
        # Extract logits from teacher
        with torch.no_grad():
            output_teacher: Seq2SeqLMOutput = self.teacher_model(**inputs)
            logits_teacher = output_teacher.logits
        
        # Soften probabilities and compute distillation loss
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * kl_div_loss(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1))
        
        # Return weighted student loss
        loss = self.args.ce_alpha * loss_ce + (1. - self.args.ce_alpha) * loss_kd
        
        return loss, output_student

    
    def _compute_loss_seq_level_k_best(self,
                                       student_model: PreTrainedModel,
                                       inputs,
                                       rank_weighting: bool = False) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels']
         
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        teacher_sequences = inputs["teacher_sequences"]  # (batch_size, num_beams, n_tokens)
        teacher_sequences_scores = inputs["teacher_sequences_scores"]  # (batch_size, num_beams)
        
        batch_size = input_features.shape[0]
        distillation_num_beams = self.args.distillation_num_beams
        n_tokens = teacher_sequences.shape[-1]
        
        # Get attention mask for teacher sequences:
        # Note that `get_attention_mask_from_tensor` expects a 2D tensor. Thus we will have to temporarily reshape `teacher_sequences`.
        teacher_sequences_attention_mask = self.get_attention_mask_from_tensor(teacher_sequences.reshape(-1, n_tokens))  # (batch_size * num_beams, n_tokens)
        teacher_sequences_attention_mask = teacher_sequences_attention_mask.reshape(batch_size, -1, n_tokens)  # (batch_size, num_beams, n_tokens)
        
        assert distillation_num_beams <= teacher_sequences.shape[1], \
            f"The number of beams for distillation must be <= the number of beams used for generation. " \
            f"Got {distillation_num_beams} beams for distillation and {teacher_sequences.shape[1]} beams for generation."
        
        # Retrieve only the beams of interest:
        teacher_sequences = teacher_sequences[:, :distillation_num_beams, :]  # (batch_size, distillation_num_beams, n_tokens)
        teacher_sequences_attention_mask = teacher_sequences_attention_mask[:, :distillation_num_beams, :]  # (batch_size, distillation_num_beams, n_tokens)
        teacher_sequences_scores = teacher_sequences_scores[:, :distillation_num_beams]  # (batch_size, distillation_num_beams)
        
        # Re-normalize scores using only the K-best sequences by applying a softmax over the beam dimension:
        teacher_sequences_prob = torch.nn.functional.softmax(teacher_sequences_scores, dim=-1)  # (batch_size, distillation_num_beams)
        
        # We want to take advantage of the fact that the batch dimension and the beam dimension are indifferent to process them all at once.
        # Important note: Technically, the student model should be able to process the entire batch of size `batch_size * distillation_num_beams`. If
        #                 this is not the case, we need to reduce the batch size accordingly.
        teacher_sequences = teacher_sequences.reshape(batch_size * distillation_num_beams, n_tokens)  # (batch_size * distillation_num_beams, n_tokens)
        teacher_sequences_attention_mask = teacher_sequences_attention_mask.reshape(batch_size * distillation_num_beams, n_tokens)  # (batch_size * distillation_num_beams, n_tokens)
        
        # Repeat input features for the number of beams. The resulting tensor will have shape (batch_size * distillation_num_beams, dim_features).
        input_features = input_features.repeat_interleave(distillation_num_beams, dim=0)  # (batch_size * distillation_num_beams, 80, 3000)
        
        # Forward pass through student:
        # Note that we excluded the EOT token "<|endoftext|>" as generation is supposed to stop here.
        student_output: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                decoder_input_ids=teacher_sequences[:, :-1],
                                                                decoder_attention_mask=teacher_sequences_attention_mask[:, :-1])
        
        
        # Extract cross-entropy loss and logits from student output:
        # TODO: Get CE wrt to label and not with respect to teacher label !!!
        # loss_ce = student_output.loss  # mean cross-entropy -> (1,)
        student_logits = student_output.logits  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        vocab_size = student_logits.shape[-1]
        
        # Temporarily reshape logits to be able to properly use softmax along the last dimension:
        student_logits = student_logits.reshape(batch_size, distillation_num_beams, n_tokens - 1, -1)  # (batch_size, distillation_num_beams, n_tokens-1, vocab_size)
        
        # Normalize logits:
        student_log_prob_all = torch.nn.functional.log_softmax(student_logits, dim=-1)  # (batch_size, distillation_num_beams, n_tokens-1, vocab_size)
        
        # Reshape back to original shape:
        student_log_prob_all = student_log_prob_all.reshape(batch_size * distillation_num_beams, n_tokens - 1, -1)  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        
        # Set the values associated to the pad tokens to 0:
        # Note: Because we will sum the log-probabilities to make use of the product rule in the log space,
        #       a sufficient method to ignore the padded values is to set them to 0.
        
        # Repeat attention_mask for the n_vocab dimension:
        student_log_prob_all_mask = teacher_sequences_attention_mask[:, :-1, None].expand(-1, -1, vocab_size)
        output_log_prob_all_masked = student_log_prob_all.masked_fill(student_log_prob_all_mask.ne(1), 0)
        
        # Get log-probabilities for each generation step:
        log_prob_t_hat_step_wise = output_log_prob_all_masked.take_along_dim(teacher_sequences[:, 1:, None], dim=-1)  # (batch_size * distillation_num_beams, n_tokens-1, 1)
        
        # Compute the sequence negative log-probability of `y_hat`, which is equal to `loss_kd`:
        # log_prob_t_hat_step_wise.squeeze() -> (batch_size * distillation_num_beams, n_tokens-1)
        # Hence we need to sum over the last dimension to get the sequence log-probability.
        loss_kd = - torch.sum(log_prob_t_hat_step_wise.squeeze(), axis=-1)  # (batch_size * distillation_num_beams,)
        loss_kd = loss_kd.reshape(batch_size, distillation_num_beams)  # (batch_size, distillation_num_beams)
        
        # Compute the weighted mean of the sequence log-probabilities:
        # Inputs:
        # - weights -> (distillation_num_beams,)
        # - teacher_sequences_prob -> (batch_size, distillation_num_beams)
        # - loss_kd -> (batch_size, distillation_num_beams)
        # Output:
        # - weights * teacher_sequences_prob * loss_kd -> (batch_size, distillation_num_beams) [broadcasting]
        if rank_weighting:
            weights = self.get_rank_based_exp_decay_weights(K=self.args.distillation_num_beams, beta=self.args.decay_beta)
            loss_kd = torch.sum(weights * teacher_sequences_prob * loss_kd, axis=-1)  # (batch_size,)
        else:
            loss_kd = torch.sum(teacher_sequences_prob * loss_kd, axis=-1)  # (batch_size,)
        
        # Because the default cross-entropy loss is computed with reduction='mean', we need to
        # also take the mean of the sequence log-probabilities to be consistent:
        loss_kd = torch.mean(loss_kd,)  # (1,)
        
        # TODO: Take the cross-entropy into account
        # # Return weighted student loss
        # loss = self.args.ce_alpha * loss_ce + (1. - self.args.ce_alpha) * loss_kd
        
        return loss_kd, student_output
    
    
    def _compute_loss_seq_level_k_best_uniform(self,
                                               student_model: PreTrainedModel,
                                               inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best uniform sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        return self._compute_loss_seq_level_k_best(student_model, inputs, rank_weighting=False)
    
    
    def _compute_loss_seq_level_k_best_ranked(self,
                                              student_model: PreTrainedModel,
                                              inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best ranked sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        return self._compute_loss_seq_level_k_best(student_model, inputs, rank_weighting=True)
    
    
    def get_attention_mask_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns the attention mask from a tensor of shape (batch_size, n_tokens).
        
        Example:
        - Input: tensor([[50257.,  50362.,     76.,    1694.,    627.,   50256.],
                         [50257.,  50362.,  13099.,   50256.,  50256.,   50256.]])
        
        - Output: tensor([[1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0]])
        
        Note: The padding token is the same as the end-of-text (EOT) token. Therefore,
              we should turn off the attention for all the EOT tokens at the exception
              of the first EOT token of each row (see example above).
        """
        
        assert tensor.ndim == 2, \
            f"The tensor must be 2D. Got {tensor.ndim} dimensions."
        
        pad_token_id = self.student_tokenizer.pad_token_id
        indices = (tensor == pad_token_id).long().argmax(dim=-1)
        attention_mask = torch.zeros_like(tensor, dtype=torch.long)
        for idx, row in zip(indices, attention_mask):
            row[idx+1:] = 1  # ignore the first EOT token of each row
        return attention_mask
    
    
    @staticmethod
    def get_rank_based_exp_decay_weights(K: int, beta: float) -> torch.Tensor:
        """
        Returns a tensor of shape (K,) containing the weights for the rank-based exponential decay.
        """
        weights = torch.zeros(K).to(device)
        for k in range(1, K + 1):
            weights[k - 1] = np.exp(- beta * (k - 1))
        weights /= torch.sum(weights)
        return weights
