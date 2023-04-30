import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TrainingArguments, Trainer, PreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArguments(TrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
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
    Trainer class for distillation. Should be used with `DistillationTrainingArguments`.
    """
    def __init__(self,
                 teacher_model: PreTrainedModel,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs,
                     return_outputs: bool=False):
        inputs = inputs.to(device)
        outputs_stu = model(**inputs)
        
        # Extract cross-entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        
        # Extract logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(  # type: ignore
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),  # type: ignore
            F.softmax(logits_tea / self.args.temperature, dim=-1))  # type: ignore
        
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd  # type: ignore
        
        return (loss, outputs_stu) if return_outputs else loss
