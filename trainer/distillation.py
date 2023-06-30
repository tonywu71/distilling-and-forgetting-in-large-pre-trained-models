from typing import Optional, Literal

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedModel, WhisperProcessor, WhisperTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.utils import get_fast_tokenizer_from_tokenizer
from trainer.trainer_utils import get_language_token


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method_distil: Literal["word_level", "seq_level_uniform", "seq_level_ranked"],
                 alpha_ce: float,
                 temperature: Optional[float] = None,
                 distillation_num_beams: Optional[int] = None,
                 beta_decay: Optional[float] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.method_distil = method_distil
        self.alpha_ce = alpha_ce
        self.temperature = temperature
        self.distillation_num_beams = distillation_num_beams
        self.beta_decay = beta_decay


class DistillationTrainer(Seq2SeqTrainer):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 student_processor: WhisperProcessor,
                 teacher_model: Optional[PreTrainedModel] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.student_processor = student_processor
        self.student_tokenizer = get_fast_tokenizer_from_tokenizer(self.student_processor.tokenizer)
        self.teacher_model = teacher_model
        
        # Sanity checks:
        if self.args.method_distil == "word_level":
            assert self.teacher_model is not None, \
                "The `teacher_model` must be set for word-level distillation."
        
        
        self.METHOD_DISTIL_TO_LOSS_FCT = {
            "word_level": self._compute_loss_word_level,
            "seq_level_uniform": self._compute_loss_seq_level_uniform,
            "seq_level_ranked": self._compute_loss_seq_level_ranked
        }
    
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        """
        Overriding `get_eval_dataloader` as we should not pass the attention mask during evaluation.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=self.student_tokenizer,
                                                             feature_extractor=self.student_processor.feature_extractor,
                                                             return_attention_mask=False)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    
    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs,
                     return_outputs: bool = False):
        """
        Computes the loss according to the distillation method specified in `self.args.method_distil`.
        """
        loss, output_student = self.METHOD_DISTIL_TO_LOSS_FCT[self.args.method_distil](student_model=model,
                                                                                       inputs=inputs,
                                                                                       teacher_model=self.teacher_model)
        return (loss, output_student) if return_outputs else loss
    
    
    def _compute_loss_word_level(self,
                                 student_model: PreTrainedModel,
                                 inputs,
                                 teacher_model: PreTrainedModel,
                                 language: Optional[str] = None) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for word-level distillation.
        """
        
        assert teacher_model is not None, \
            "The `teacher_model` must be set for word-level distillation."
        
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask']
        
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask = inputs["attention_mask"]  # (batch_size, n_tokens_labels)
        
        # [For TAC] Replace the language token with the new token language if specified:
        if language:
            labels[:, 1] = get_language_token(language)  # hardcoded: 1 is the position of the language token
        
        # Forward pass through student (teacher-forced):
        output_student: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                labels=labels,
                                                                attention_mask=attention_mask)
        logits_student = output_student.logits  # (batch_size, n_tokens_labels, vocab_size)
        
        # Compute the cross-entropy loss:
        loss_ce = output_student.loss  # (1,)
        
        # Extract logits from teacher
        with torch.no_grad():
            # Forward pass through teacher (teacher-forced):
            output_teacher: Seq2SeqLMOutput = teacher_model.forward(input_features=input_features,
                                                                    labels=labels,
                                                                    attention_mask=attention_mask)
            logits_teacher = output_teacher.logits  # (batch_size, n_tokens_labels, vocab_size)
        
        # Initialize KL-divergence loss:
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Important:
        # - `KLDivLoss` argument order is the opposite of the one for the KL(·||·) mathematical notation
        # - `KLDivLoss` expects log-probabilities for `input` to avoid underflow issues
        
        # Soften probabilities and compute distillation loss:
        # NOTE: `input` should be log-probabilities according to the documentation of `KLDivLoss`.
        loss_kd = self.args.temperature ** 2 * kl_div_loss(
            input=F.log_softmax(logits_student / self.args.temperature, dim=-1),
            target=F.softmax(logits_teacher / self.args.temperature, dim=-1))  # (1,)
        
        # Return weighted student loss
        loss = self.args.alpha_ce * loss_ce + (1. - self.args.alpha_ce) * loss_kd  # (1,)
        
        return loss, output_student

    
    def _compute_loss_seq_level_k_best(self,
                                       student_model: PreTrainedModel,
                                       inputs,
                                       teacher_model: PreTrainedModel = None,
                                       language: Optional[str] = None,
                                       rank_weighting: bool = False) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best sequence-level distillation where `k = self.args.distillation_num_beams`.
        """
        
        # If `language` is not specified, use the language of the tokenizer:
        if language is None:
            language = self.student_tokenizer.language
            assert language is not None, "The `language` must be specified in the tokenizer."
        
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask', 'teacher_sequences',
                                    #                 'attention_mask_teacher_sequences', 'teacher_sequences_scores']
        
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask_labels = inputs["attention_mask"]  # (batch_size, n_tokens_labels)
        teacher_sequences = inputs["teacher_sequences"]  # (batch_size, num_beams, n_tokens_teacher_seq)
        attention_mask_teacher_sequences = inputs["attention_mask_teacher_sequences"]  # (batch_size, num_beams, n_tokens_teacher_seq)
        teacher_sequences_scores = inputs["teacher_sequences_scores"]  # (batch_size, num_beams)
        
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
        labels_right_shifted = shift_tokens_right(labels, student_model.config.pad_token_id, student_model.config.decoder_start_token_id)
        student_output_wrt_labels: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                           decoder_input_ids=labels_right_shifted,
                                                                           decoder_attention_mask=attention_mask_labels,
                                                                           labels=labels)
        
        # NOTE: `shift_tokens_right` already added the BOS token and replaced the loss mask token with the pad token, so we can safely use the sequence as decoder input.
        
        # Compute the cross-entropy loss:
        loss_ce = student_output_wrt_labels.loss  # (1,)
        
        # Repeat input features for the number of beams. The resulting tensor will have shape (batch_size * distillation_num_beams, dim_features).
        input_features = input_features.repeat_interleave(distillation_num_beams, dim=0)  # (batch_size * distillation_num_beams, 80, 3000)
        
        # Forward pass through student with respect to teacher sequences as decoder input:
        teacher_sequences_right_shifted = shift_tokens_right(teacher_sequences, student_model.config.pad_token_id, student_model.config.decoder_start_token_id)
        student_output_wrt_teacher: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
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
        student_log_prob_all_mask = attention_mask_teacher_sequences.expand(-1, -1, vocab_size)  # right-shifted -> (batch_size * distillation_num_beams, n_tokens_teacher_seq, vocab_size)
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
        if rank_weighting:
            weights = self.get_rank_based_exp_decay_weights(K=self.args.distillation_num_beams, beta=self.args.beta_decay)
            loss_kd = torch.sum(weights * teacher_sequences_prob * loss_kd_per_sentence, axis=-1)  # (batch_size,)
        else:
            loss_kd = torch.sum(teacher_sequences_prob * loss_kd_per_sentence, axis=-1)  # (batch_size,)
        
        # Because the default cross-entropy loss is computed with reduction='mean', we need to
        # also take the mean of the sequence log-probabilities to be consistent:
        loss_kd = torch.mean(loss_kd,)  # (1,)
        
        # Return weighted student loss
        loss = self.args.alpha_ce * loss_ce + (1. - self.args.alpha_ce) * loss_kd
        
        return loss, student_output_wrt_teacher
    
    
    def _compute_loss_seq_level_uniform(self,
                                               student_model: PreTrainedModel,
                                               inputs,
                                               teacher_model: PreTrainedModel = None,
                                               language: str = "english") -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best uniform sequence-level distillation where `k = self.args.distillation_num_beams`.
        
        Note: `teacher_model` should be set to `None` when using this method. It is kept as an argument only for consistency.
        """
        return self._compute_loss_seq_level_k_best(student_model, inputs, language=language, rank_weighting=False)
    
    
    def _compute_loss_seq_level_ranked(self,
                                              student_model: PreTrainedModel,
                                              inputs,
                                              teacher_model: PreTrainedModel = None,
                                              language: str = "english") -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        """
        Compute the loss for k-best ranked sequence-level distillation where `k = self.args.distillation_num_beams`.
        
        Note: `teacher_model` should be set to `None` when using this method. It is kept as an argument only for consistency.
        """
        return self._compute_loss_seq_level_k_best(student_model, inputs, language=language, rank_weighting=True)
    
    
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
