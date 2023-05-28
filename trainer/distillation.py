from typing import Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TrainingArguments, Trainer, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from utils.constants import GEN_MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(TrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method: Literal["word_level", "seq_level_mode", "seq_level_k_best_uniform", "seq_level_k_best_ranked"],
                 ce_alpha: float,
                 temperature: Optional[float]=None,
                 distillation_num_beams: Optional[int] = None,
                 decay_beta: Optional[float]=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.ce_alpha = ce_alpha
        self.temperature = temperature
        self.distillation_num_beams = distillation_num_beams
        self.decay_beta = decay_beta


class DistillationTrainer(Trainer):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 teacher_model: PreTrainedModel,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        
        self.METHOD_TO_LOSS_FCT = {
            "word_level": self._compute_loss_word_level,
            "seq_level_mode": self._compute_loss_seq_level_mode,
            "seq_level_k_best_uniform": self._compute_loss_seq_level_k_best_uniform,
            "seq_level_k_best_ranked": self._compute_loss_seq_level_k_best_ranked
        }
    
    
    def compute_loss(self,
                     student_model: PreTrainedModel,
                     inputs,
                     return_outputs: bool=False):
        loss, output_student = self.METHOD_TO_LOSS_FCT[self.args.method](student_model, inputs)
        return (loss, output_student) if return_outputs else loss
    
    
    def _compute_loss_word_level(self,
                                 student_model: PreTrainedModel,
                                 inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
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


    def _compute_loss_seq_level_mode(self,
                                     student_model: PreTrainedModel,
                                     inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels']
        input_features = inputs["input_features"]
        
        # Generate teacher predictions using K-beam search:
        pred_ids_teacher = self.teacher_model.generate(input_features,
                                                       max_length=GEN_MAX_LENGTH,
                                                       num_beams=self.args.distillation_num_beams)  # (batch_size, n_tokens)
        
        # Forward pass through student:
        # Note that we excluded the EOT token "<|endoftext|>" as generation is supposed to stop here.
        output_student: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                decoder_input_ids=pred_ids_teacher[:, :-1],
                                                                labels=pred_ids_teacher[:, 1:])
        
        # Extract cross-entropy loss and logits from student output:
        loss_ce = output_student.loss
        logits_student = output_student.logits  # (batch_size, n_tokens-1, vocab_size)
        
        # Normalize logits:
        log_prob_all = torch.nn.functional.log_softmax(logits_student, dim=-1)
        
        # Get log-probabilities for each generation step:
        log_prob_t_hat_step_wise = log_prob_all.take_along_dim(pred_ids_teacher[:, 1:, None], dim=-1)
        
        # Compute the sequence log-probability:
        loss_kd = (-1) * torch.sum(log_prob_t_hat_step_wise)
        
        # Return weighted student loss
        loss = self.args.ce_alpha * loss_ce + (1. - self.args.ce_alpha) * loss_kd
        
        return loss, output_student


    def _compute_loss_seq_level_k_best(self,
                                       student_model: PreTrainedModel,
                                       inputs,
                                       rank_weighting: bool=False) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        # Move inputs to device:
        inputs = inputs.to(device)  # inputs.keys -> ['input_features', 'labels']
        input_features = inputs["input_features"]
        
        # Generate teacher predictions using K-beam search:
        dict_pred_ids_teacher_beam_search = self.teacher_model.generate(input_features,
                                                                   max_length=GEN_MAX_LENGTH,
                                                                   num_beams=self.args.distillation_num_beams,
                                                                   num_return_sequences=self.args.distillation_num_beams,
                                                                   output_scores=True,
                                                                   return_dict_in_generate=True)
        
        # Retrieve the K-best tokenized sequences:
        pred_ids_teacher_beam_search = dict_pred_ids_teacher_beam_search.sequences  # (batch_size * distillation_num_beams, n_tokens)
        
        # Normalize scores using only the K-best sequences:
        prob_pred_teacher_sequences = torch.nn.functional.softmax(dict_pred_ids_teacher_beam_search.sequences_scores, dim=-1)
        
        # Note: `pred_ids_teacher` is the result of a concatenation of generate outputs along the batch dimension (axis=0).
        # Therefore, if `distillation_num_beams` = 2, then the first `batch_size` rows correspond to the 2 best beam-search predictions
        # for the first input, the next `batch_size` rows correspond to the 2 best predictions for the second input, etc...
        
        # Split `pred_ids_teacher` into `distillation_num_beams` tensors of size `(batch_size, n_tokens)`:
        pred_ids_teacher_list = torch.split(pred_ids_teacher_beam_search,  # type: ignore
                                            split_size_or_sections=input_features.shape[0],
                                            dim=0)
        
        # Placeholders for the loss terms:
        list_loss_ce: List[torch.Tensor] = []
        list_loss_kd: List[torch.Tensor] = []
        
        for pred_ids_teacher in pred_ids_teacher_list:
            # Forward pass through student:
            # Note that we excluded the EOT token "<|endoftext|>" as generation is supposed to stop here.
            output_student: Seq2SeqLMOutput = student_model.forward(input_features=input_features,
                                                                    decoder_input_ids=pred_ids_teacher[:, :-1],
                                                                    labels=pred_ids_teacher[:, 1:])
            
            # Extract cross-entropy loss and logits from student output:
            list_loss_ce.append(output_student.loss)
            logits_student = output_student.logits  # (batch_size, n_tokens-1, vocab_size)
            
            # Normalize logits:
            log_prob_all = torch.nn.functional.log_softmax(logits_student, dim=-1)
            
            # Get log-probabilities for each generation step:
            log_prob_t_hat_step_wise = log_prob_all.take_along_dim(pred_ids_teacher[:, 1:, None], dim=-1)
            
            # Compute the sequence log-probability:
            list_loss_kd.append((-1) * torch.sum(log_prob_t_hat_step_wise))
        
        
        loss_ce = torch.sum(torch.stack(list_loss_ce))
        
        if rank_weighting:
            weights = self.get_rank_based_exp_decay_weights(K=self.args.distillation_num_beams, beta=self.args.decay_beta)
            loss_kd = torch.sum(weights * prob_pred_teacher_sequences * torch.stack(list_loss_kd))
        else:
            loss_kd = torch.sum(prob_pred_teacher_sequences * torch.stack(list_loss_kd))
        
        # Return weighted student loss
        loss = self.args.ce_alpha * loss_ce + (1. - self.args.ce_alpha) * loss_kd
        
        return loss, output_student
    
    
    def _compute_loss_seq_level_k_best_uniform(self,
                                               student_model: PreTrainedModel,
                                               inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        return self._compute_loss_seq_level_k_best(student_model, inputs, rank_weighting=False)
    
    
    def _compute_loss_seq_level_k_best_ranked(self,
                                              student_model: PreTrainedModel,
                                              inputs) -> tuple[torch.Tensor, Seq2SeqLMOutput]:
        return self._compute_loss_seq_level_k_best(student_model, inputs, rank_weighting=True)
    
    
    @staticmethod
    def get_rank_based_exp_decay_weights(K: int, beta: float):
        weights = torch.zeros(K)
        for k in range(1, K + 1):
            weights[k - 1] = torch.exp(-beta * (k - 1))  # type: ignore
        weights /= torch.sum(weights)
        return weights
    