from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TrainingArguments, Trainer, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(TrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method: Literal["word_level", "seq_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"],
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
