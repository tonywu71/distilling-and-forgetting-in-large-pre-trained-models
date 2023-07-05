import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from trainer.distillation import DistillationTrainingArgumentsBase, DistillationTrainerBase


class DistillationWordLevelTrainingArguments(DistillationTrainingArgumentsBase):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 alpha_ce: float = 0.5,
                 temperature: float = 2.0,
                 *args,
                 **kwargs):
        # Drop the `method_distil` argument since it is not needed for sequence-level distillation:
        method_distil = kwargs.pop("method_distil", None)
        assert method_distil == "word_level", "method_distil should be `word_level`"
        
        super().__init__(method_distil=method_distil, *args, **kwargs)
        self.alpha_ce = alpha_ce
        self.temperature = temperature
    


class DistillationWordLevelTrainer(DistillationTrainerBase):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 args: DistillationWordLevelTrainingArguments,
                 student_processor: WhisperProcessor,
                 teacher_model: WhisperForConditionalGeneration,
                 **kwargs):
        super().__init__(args=args,
                         student_processor=student_processor,
                         **kwargs)
        self.args = args
        self.teacher_model = teacher_model
    
    
    def compute_loss(self,
                     model: WhisperForConditionalGeneration,
                     inputs,
                     return_outputs: bool = False):
        """
        Compute the loss for word-level distillation.
        """
        
        # Move inputs to device:
        inputs = inputs.to(self.device)  # inputs.keys -> ['input_features', 'labels', 'attention_mask']
        
        input_features = inputs["input_features"]  # (batch_size, 80, 3000)
        labels = inputs["labels"]  # (batch_size, n_tokens_labels)
        attention_mask = inputs["attention_mask"]  # (batch_size, n_tokens_labels)
        
        # Forward pass through student (teacher-forced):
        output_student: Seq2SeqLMOutput = model.forward(input_features=input_features,
                                                        labels=labels,
                                                        attention_mask=attention_mask)
        logits_student = output_student.logits  # (batch_size, n_tokens_labels, vocab_size)
        
        # Compute the cross-entropy loss:
        loss_ce = output_student.loss  # (1,)
        
        # Extract logits from teacher
        with torch.no_grad():
            # Forward pass through teacher (teacher-forced):
            output_teacher: Seq2SeqLMOutput = self.teacher_model.forward(input_features=input_features,
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
        
        return (loss, output_student) if return_outputs else loss
