from typing import Optional, Dict, Any

import numpy as np
import torch

from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import WhisperProcessor
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from trainer.distillation import DistillationTrainingArgumentsBase, DistillationTrainerBase


class DistillationSeqLevelTrainingArguments(DistillationTrainingArgumentsBase):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 alpha_ce: float = 0.5,
                 distillation_num_beams: int = 1,
                 beta_decay: Optional[float] = 2.0,
                 *args,
                 **kwargs):
        # Drop the `method_distil` argument since it is not needed for sequence-level distillation:
        method_distil = kwargs.pop("method_distil", None)
        assert method_distil in ["seq_level_uniform", "seq_level_ranked"], \
            "method_distil should be `seq_level_uniform` or `seq_level_ranked`"
        
        super().__init__(method_distil=method_distil,
                         *args,
                         **kwargs)
        self.alpha_ce = alpha_ce
        self.distillation_num_beams = distillation_num_beams
        self.use_ranking = (method_distil == "seq_level_ranked")
        self.beta_decay = beta_decay



class DistillationSeqLevelTrainer(DistillationTrainerBase):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 args: DistillationSeqLevelTrainingArguments,
                 student_processor: WhisperProcessor,
                 **kwargs):
        super().__init__(args=args,
                         student_processor=student_processor,
                         **kwargs)
        self.args = args
        self.counter = 0
    
    
    def compute_loss(self,
                     model: WhisperForConditionalGeneration,
                     inputs,
                     return_outputs: bool = False):
        """
        Compute the loss for sequence-level distillation. Override the `compute_loss` method of `Seq2SeqTrainer`.
        """
        if self.args.distillation_num_beams == 1:
            loss_fn = self.compute_loss_1_best
        else:
            loss_fn = self.compute_loss_k_best
        return loss_fn(model, inputs, return_outputs)
    
    
    def compute_loss_1_best(self,
                            model: WhisperForConditionalGeneration,
                            inputs: Dict[str, Any],
                            return_outputs: bool = False):
        """
        Compute the loss for 1-best sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        inputs_ce = inputs.copy()
        inputs_ce.pop("teacher_sequences")
        inputs_ce.pop("attention_mask_teacher_sequences")
        
        loss_ce, output_student_wrt_labels = super().compute_loss(model, inputs_ce, return_outputs=True)
        
        inputs_kd = inputs.copy()
        inputs_kd["labels"] = inputs["teacher_sequences"]
        inputs_kd["attention_mask"] = inputs["attention_mask_teacher_sequences"]
        inputs_kd.pop("teacher_sequences")
        inputs_kd.pop("attention_mask_teacher_sequences")
        
        loss_kd = super().compute_loss(model, inputs_kd, return_outputs=False)
        
        loss = self.args.alpha_ce * loss_ce + (1 - self.args.alpha_ce) * loss_kd
        return (loss, output_student_wrt_labels) if return_outputs else loss
    
    
    def compute_loss_k_best(self,
                            model: WhisperForConditionalGeneration,
                            inputs: Dict[str, Any],
                            return_outputs: bool = False):
        """
        Compute the loss for k-best sequence-level distillation where `k = self.args.distillation_num_beams`.
        If k = 1, use `compute_loss_1_best` instead.
        """
        
        language = self.student_tokenizer.language
        assert language is not None, "The `language` must be specified in the tokenizer."
        
        # Move inputs to device:
        inputs = inputs.to(self.device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask', 'teacher_sequences',
                                         #                 'attention_mask_teacher_sequences', 'teacher_sequences_scores']
        
        # Get all useful features from `inputs`:
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask_labels = inputs["attention_mask"]  # (batch_size, n_tokens_labels)
        teacher_sequences = inputs["teacher_sequences"]  # (batch_size, num_beams, n_tokens_teacher_seq)
        attention_mask_teacher_sequences = inputs["attention_mask_teacher_sequences"]  # (batch_size, num_beams, n_tokens_teacher_seq)
        teacher_sequences_scores = inputs["teacher_sequences_scores"]  # (batch_size, num_beams)
        
        # Get additional useful information for K-best KD:
        batch_size = input_features.shape[0]
        distillation_num_beams = self.args.distillation_num_beams
        n_tokens_teacher_seq = teacher_sequences.shape[-1]
        
        # Re-normalize scores using only the K-best sequences by applying a softmax over the beam dimension:
        teacher_sequences_prob = torch.nn.functional.softmax(teacher_sequences_scores, dim=-1)  # (batch_size, distillation_num_beams)
        
        # We want to take advantage of the fact that the batch dimension and the beam dimension are indifferent to process them all at once.
        # Important note: Technically, the student model should be able to process the entire batch of size `batch_size * distillation_num_beams`. If
        #                 this is not the case, we need to reduce the batch size accordingly.
        teacher_sequences = teacher_sequences.reshape(batch_size * distillation_num_beams, n_tokens_teacher_seq)  # (batch_size * distillation_num_beams, n_tokens_teacher_seq)
        attention_mask_teacher_sequences = attention_mask_teacher_sequences.reshape(batch_size * distillation_num_beams, n_tokens_teacher_seq)  # (batch_size * distillation_num_beams, n_tokens_teacher_seq)
        
        
        # Forward pass through student with labels as decoder input:
        labels_right_shifted = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
        student_output_wrt_labels: Seq2SeqLMOutput = model.forward(input_features=input_features,
                                                                   decoder_input_ids=labels_right_shifted,
                                                                   decoder_attention_mask=attention_mask_labels,
                                                                   labels=labels)
        
        # NOTE: `shift_tokens_right` already added the BOS token and replaced the loss mask token with the pad token, so we can safely use the sequence as decoder input.
        
        # Compute the cross-entropy loss:
        loss_ce = student_output_wrt_labels.loss  # (1,)
        
        # Repeat input features for the number of beams. The resulting tensor will have shape (batch_size * distillation_num_beams, dim_features).
        input_features = input_features.repeat_interleave(distillation_num_beams, dim=0)  # (batch_size * distillation_num_beams, 80, 3000)
        
        # Forward pass through student with respect to teacher sequences as decoder input:
        teacher_sequences_right_shifted = shift_tokens_right(teacher_sequences, model.config.pad_token_id, model.config.decoder_start_token_id)
        student_output_wrt_teacher: Seq2SeqLMOutput = model.forward(input_features=input_features,
                                                                    decoder_input_ids=teacher_sequences_right_shifted,
                                                                    decoder_attention_mask=attention_mask_teacher_sequences)
        
        student_logits_wrt_teacher = student_output_wrt_teacher.logits  # (batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        _, _, vocab_size = student_logits_wrt_teacher.shape
        
        # Temporarily reshape logits to be able to properly use softmax along the last dimension:
        student_logits = student_logits_wrt_teacher.reshape(batch_size, distillation_num_beams, n_tokens_teacher_seq, vocab_size)  # (batch_size, distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        
        # Normalize logits:
        student_log_prob_all = torch.nn.functional.log_softmax(student_logits, dim=-1)  # (batch_size, distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        
        # Reshape back to original shape:
        student_log_prob_all = student_log_prob_all.reshape(batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)  # (batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        
        # NOTE: The first `K = distillation_num_beams` rows of output_log_prob_all_masked correspond to the same example but with different beams.
        
        # Set the values associated to the pad tokens to 0:
        # NOTE: Because we will sum the log-probabilities to make use of the product rule in the log space,
        #       a sufficient method to ignore the padded values is to set them to 0.
        
        # Repeat attention_mask for the n_vocab dimension:
        student_log_prob_all_mask = attention_mask_teacher_sequences[:, :, None].expand(-1, -1, vocab_size)  # right-shifted -> (batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        output_log_prob_all_masked = student_log_prob_all.masked_fill(student_log_prob_all_mask.ne(1), 0)  # (batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)
        
        # Get log-probabilities for each generation step:
        log_prob_t_hat_step_wise = output_log_prob_all_masked.take_along_dim(teacher_sequences_right_shifted[:, :, None], dim=-1)  # (batch_size * distillation_num_beams, n_tokens_teacher_seq, 1)
        
        # Compute the sequence negative log-probability of `y_hat`, which is equal to `loss_kd`:
        # log_prob_t_hat_step_wise.squeeze() -> (batch_size * distillation_num_beams, n_tokens_teacher_seq)
        # Hence we need to sum over the last dimension to get the sequence log-probability (product rule in the log space):
        loss_kd_per_sentence = - torch.sum(log_prob_t_hat_step_wise.squeeze(), axis=-1)  # (batch_size * distillation_num_beams,)
        loss_kd_per_sentence = loss_kd_per_sentence.reshape(batch_size, distillation_num_beams)  # (batch_size, distillation_num_beams)
        
        # Compute the weighted mean of the sequence log-probabilities:
        # Inputs:
        # - weights -> (distillation_num_beams,)
        # - teacher_sequences_prob -> (batch_size, distillation_num_beams)
        # - loss_kd_per_sentence -> (batch_size, distillation_num_beams)
        # Output:
        # - weights * teacher_sequences_prob * loss_kd_per_sentence -> (batch_size, distillation_num_beams) [broadcasting]

        if self.args.use_ranking:
            weights = self.get_rank_based_exp_decay_weights(K=self.args.distillation_num_beams, beta=self.args.beta_decay)
            breakpoint()
            loss_kd = torch.sum(weights * teacher_sequences_prob * loss_kd_per_sentence, axis=-1)  # (batch_size,)
        else:
            loss_kd = torch.sum(teacher_sequences_prob * loss_kd_per_sentence, axis=-1)  # (batch_size,)
        
        # Because the default cross-entropy loss is computed with reduction='mean', we need to
        # also take the mean of the sequence log-probabilities to be consistent:
        loss_kd = torch.mean(loss_kd,)  # (1,)
        
        # Return weighted student loss
        loss = self.args.alpha_ce * loss_ce + (1. - self.args.alpha_ce) * loss_kd
        
        return (loss, student_output_wrt_labels) if return_outputs else loss
    
    
    def get_rank_based_exp_decay_weights(self, K: int, beta: float) -> torch.Tensor:
        """
        Returns a tensor of shape (K,) containing the weights for the rank-based exponential decay.
        """
        assert beta > 0, f"The `beta` parameter must be > 0. Got `{beta}`."
        
        weights = torch.zeros(K).to(self.device)
        for k in range(1, K + 1):
            weights[k - 1] = np.exp(- beta * (k - 1))
        weights /= torch.sum(weights)
        
        return weights
