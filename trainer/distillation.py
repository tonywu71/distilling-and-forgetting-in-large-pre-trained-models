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
                 alpha: float=0.5,
                 temperature: float=2.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


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
    
    
    def compute_loss(self,
                     student_model: PreTrainedModel,
                     inputs,
                     return_outputs: bool=False):
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
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(  # type: ignore
            F.log_softmax(logits_student / self.args.temperature, dim=-1),  # type: ignore
            F.softmax(logits_teacher / self.args.temperature, dim=-1))  # type: ignore
        
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd  # type: ignore
        
        return (loss, output_student) if return_outputs else loss
