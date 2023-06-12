from typing import Optional, Literal

import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TrainingArguments, Trainer, PreTrainedModel, WhisperTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from trainer.prompting import get_labels_with_prompt, get_attention_mask_with_prompt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(TrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method: Literal["word_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"],
                 alpha_ce: float,
                 temperature: Optional[float] = None,
                 distillation_num_beams: Optional[int] = None,
                 beta_decay: Optional[float] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.alpha_ce = alpha_ce
        self.temperature = temperature
        self.distillation_num_beams = distillation_num_beams
        self.beta_decay = beta_decay


class DistillationTrainer(Trainer):
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
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask_labels']
        
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask_labels = inputs["attention_mask_labels"]  # (batch_size, n_tokens_labels)
        
        # Add prompt to labels and attention mask:
        labels_with_prompt, n_prefix_tokens, n_suffix_tokens = get_labels_with_prompt(labels,
                                                                                      tokenizer=self.student_tokenizer,
                                                                                      language="en",
                                                                                      task="transcribe",
                                                                                      no_timestamps=True)
        attention_mask_labels_with_prompt = get_attention_mask_with_prompt(attention_mask_labels,
                                                                           n_prefix_tokens=n_prefix_tokens,
                                                                           n_suffix_tokens=n_suffix_tokens)
        
        
        # Forward pass through student:
        output_student: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                decoder_input_ids=labels_with_prompt[:, :-1],  # don't predict when current token is EOS
                                                                decoder_attention_mask=attention_mask_labels_with_prompt[:, :-1])
        logits_student = output_student.logits  # (batch_size, n_tokens_labels, vocab_size)
        
        # Compute the cross-entropy loss:
        loss_ce = self.compute_cross_entropy_loss(output_student=output_student,
                                                  labels_with_prompt=labels_with_prompt,
                                                  n_prefix_tokens=n_prefix_tokens)
        
        # Extract logits from teacher
        with torch.no_grad():
            output_teacher: Seq2SeqLMOutput = self.teacher_model.forward(input_features=input_features,
                                                                         decoder_input_ids=labels_with_prompt[:, :-1],  # don't predict when current token is EOS
                                                                         decoder_attention_mask=attention_mask_labels_with_prompt[:, :-1])
            logits_teacher = output_teacher.logits
        
        # Initialize KL-divergence loss:
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Important:
        # - `KLDivLoss` argument order is the opposite of the one for the KL(·||·) methematical notation
        # - `KLDivLoss` expects log-probabilities for `input` to avoid underflow issues
        
        # Soften probabilities and compute distillation loss:
        # Note: `input` should be log-probabilities according to the documentation of `KLDivLoss`.
        loss_kd = self.args.temperature ** 2 * kl_div_loss(
            input=F.log_softmax(logits_student / self.args.temperature, dim=-1),
            target=F.softmax(logits_teacher / self.args.temperature, dim=-1))
        
        # Return weighted student loss
        loss = self.args.alpha_ce * loss_ce + (1. - self.args.alpha_ce) * loss_kd
        
        return loss, output_student

    
    def _compute_loss_seq_level_k_best(self,
                                       student_model: PreTrainedModel,
                                       inputs,
                                       rank_weighting: bool = False) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask_labels', 'teacher_sequences',
                                    #                 'attention_mask_teacher_sequences', 'teacher_sequences_scores']
        
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask_labels = inputs["attention_mask_labels"]  # (batch_size, n_tokens_labels)
        teacher_sequences = inputs["teacher_sequences"]  # (batch_size, num_beams, n_tokens)
        attention_mask_teacher_sequences = inputs["attention_mask_teacher_sequences"]  # (batch_size, num_beams, n_tokens)
        teacher_sequences_scores = inputs["teacher_sequences_scores"]  # (batch_size, num_beams)
        
        batch_size = input_features.shape[0]
        distillation_num_beams = self.args.distillation_num_beams
        n_tokens = teacher_sequences.shape[-1]
        
        # Sanity check:
        assert distillation_num_beams <= teacher_sequences.shape[1], \
            f"The number of beams for distillation must be <= the number of beams used for generation. " \
            f"Got {distillation_num_beams} beams for distillation and {teacher_sequences.shape[1]} beams for generation."
        
        # Retrieve only the beams of interest:
        teacher_sequences = teacher_sequences[:, :distillation_num_beams, :]  # (batch_size, distillation_num_beams, n_tokens)
        attention_mask_teacher_sequences = attention_mask_teacher_sequences[:, :distillation_num_beams, :]  # (batch_size, distillation_num_beams, n_tokens)
        teacher_sequences_scores = teacher_sequences_scores[:, :distillation_num_beams]  # (batch_size, distillation_num_beams)
        
        # Re-normalize scores using only the K-best sequences by applying a softmax over the beam dimension:
        teacher_sequences_prob = torch.nn.functional.softmax(teacher_sequences_scores, dim=-1)  # (batch_size, distillation_num_beams)
        
        # We want to take advantage of the fact that the batch dimension and the beam dimension are indifferent to process them all at once.
        # Important note: Technically, the student model should be able to process the entire batch of size `batch_size * distillation_num_beams`. If
        #                 this is not the case, we need to reduce the batch size accordingly.
        teacher_sequences = teacher_sequences.reshape(batch_size * distillation_num_beams, n_tokens)  # (batch_size * distillation_num_beams, n_tokens)
        attention_mask_teacher_sequences = attention_mask_teacher_sequences.reshape(batch_size * distillation_num_beams, n_tokens)  # (batch_size * distillation_num_beams, n_tokens)
        
        # Add prompt to labels and attention mask:
        labels_with_prompt, n_prefix_tokens_labels, n_suffix_tokens_labels = get_labels_with_prompt(labels,
                                                                                                    tokenizer=self.student_tokenizer,
                                                                                                    language="en",
                                                                                                    task="transcribe",
                                                                                                    no_timestamps=True)
        attention_mask_labels_with_prompt = get_attention_mask_with_prompt(attention_mask_labels,
                                                                           n_prefix_tokens=n_prefix_tokens_labels,
                                                                           n_suffix_tokens=n_suffix_tokens_labels)
        
        # Forward pass through student with labels as decoder input:
        student_output_wrt_labels: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                           decoder_input_ids=labels_with_prompt[:, :-1],
                                                                           decoder_attention_mask=attention_mask_labels_with_prompt[:, :-1])
        
        # Compute the cross-entropy loss:
        loss_ce = self.compute_cross_entropy_loss(output_student=student_output_wrt_labels,
                                                  labels_with_prompt=labels_with_prompt,
                                                  n_prefix_tokens=n_prefix_tokens_labels)
        
        # Repeat input features for the number of beams. The resulting tensor will have shape (batch_size * distillation_num_beams, dim_features).
        input_features = input_features.repeat_interleave(distillation_num_beams, dim=0)  # (batch_size * distillation_num_beams, 80, 3000)
        
        # Add prompt to labels and attention mask:
        teacher_sequences_with_prompt, n_prefix_tokens_teacher_seq, n_suffix_tokens_teacher_seq = get_labels_with_prompt(teacher_sequences,
                                                                                                                         tokenizer=self.student_tokenizer,
                                                                                                                         language="en",
                                                                                                                         task="transcribe",
                                                                                                                         no_timestamps=True)
        attention_mask_teacher_sequences_with_prompt = get_attention_mask_with_prompt(attention_mask_labels,
                                                                                      n_prefix_tokens=n_prefix_tokens_teacher_seq,
                                                                                      n_suffix_tokens=n_suffix_tokens_teacher_seq)
        
        # Forward pass through student:
        student_output_wrt_teacher: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                            decoder_input_ids=teacher_sequences_with_prompt[:, :-1],
                                                                            decoder_attention_mask=attention_mask_teacher_sequences_with_prompt[:, :-1])
        
        student_logits = student_output_wrt_teacher.logits  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        vocab_size = student_logits.shape[-1]
        
        # Temporarily reshape logits to be able to properly use softmax along the last dimension:
        student_logits = student_logits.reshape(batch_size, distillation_num_beams, n_tokens - 1, -1)  # (batch_size, distillation_num_beams, n_tokens-1, vocab_size)
        
        # Normalize logits:
        student_log_prob_all = torch.nn.functional.log_softmax(student_logits, dim=-1)  # (batch_size, distillation_num_beams, n_tokens-1, vocab_size)
        
        # Reshape back to original shape:
        student_log_prob_all = student_log_prob_all.reshape(batch_size * distillation_num_beams, n_tokens - 1, -1)  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        
        # Note: The first `K = distillation_num_beams` rows of output_log_prob_all_masked correspond to the same example but with different beams.
        
        # Set the values associated to the pad tokens to 0:
        # Note: Because we will sum the log-probabilities to make use of the product rule in the log space,
        #       a sufficient method to ignore the padded values is to set them to 0.
        
        # Repeat attention_mask for the n_vocab dimension:
        student_log_prob_all_mask = attention_mask_teacher_sequences[:, :-1, None].expand(-1, -1, vocab_size)  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        output_log_prob_all_masked = student_log_prob_all.masked_fill(student_log_prob_all_mask.ne(1), 0)  # (batch_size * distillation_num_beams, n_tokens-1, vocab_size)
        
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
            weights = self.get_rank_based_exp_decay_weights(K=self.args.distillation_num_beams, beta=self.args.beta_decay)
            loss_kd = torch.sum(weights * teacher_sequences_prob * loss_kd, axis=-1)  # (batch_size,)
        else:
            loss_kd = torch.sum(teacher_sequences_prob * loss_kd, axis=-1)  # (batch_size,)
        
        # Because the default cross-entropy loss is computed with reduction='mean', we need to
        # also take the mean of the sequence log-probabilities to be consistent:
        loss_kd = torch.mean(loss_kd,)  # (1,)
        
        # Return weighted student loss
        loss = self.args.alpha_ce * loss_ce + (1. - self.args.alpha_ce) * loss_kd
        
        return loss, student_output_wrt_teacher
    
    
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
    

    def compute_cross_entropy_loss(self,
                                   output_student: Seq2SeqLMOutput,
                                   labels_with_prompt: torch.Tensor,
                                   n_prefix_tokens: int) -> torch.Tensor:
        # Remove what the model tried to predict for the special tokens at the beginning of the sequence:
        logits_student = output_student.logits[:, n_prefix_tokens-1:, :]
        
        # To be used with categorical targets, `F.cross_entropy` needs to be used with a tensor for which the 2nd dimension is the class dimension.
        # See https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html for more information.
        logits_student = rearrange(logits_student, "b s v -> b v s")  # (batch_size, vocab_size, n_tokens_labels)
        
        # Compute cross-entropy loss:
        loss_ce = F.cross_entropy(input=logits_student,
                                  target=labels_with_prompt[:, n_prefix_tokens:],
                                  ignore_index=self.student_tokenizer.pad_token_id)  # (1,)
        return loss_ce
    
    
    @staticmethod
    def get_rank_based_exp_decay_weights(K: int, beta: float) -> torch.Tensor:
        """
        Returns a tensor of shape (K,) containing the weights for the rank-based exponential decay.
        """
        assert beta > 0, f"The `beta` parameter must be > 0. Got `{beta}`."
        
        weights = torch.zeros(K).to(device)
        for k in range(1, K + 1):
            weights[k - 1] = np.exp(- beta * (k - 1))
        weights /= torch.sum(weights)
        
        return weights
