from undecorated import undecorated
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, PreTrainedModel

from trainer.distillation import DistillationTrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordLevelDistillationTrainingArguments(DistillationTrainingArguments):
    """
    Subclass of `DistillationTrainingArguments` used for word-level distillation.
    Note: Same behavior as `DistillationTrainingArguments` but kept for consistency.
    """
    def __init__(self,
                 alpha: float=0.5,
                 temperature: float=2.0,
                 *args,
                 **kwargs):
        super().__init__(ce_alpha=alpha,
                         temperature=temperature,
                         *args,
                         **kwargs)


class SeqLevelDistillationTrainingArguments(DistillationTrainingArguments):
    """
    Subclass of `DistillationTrainingArguments` used for sequence-level distillation.
    """
    def __init__(self,
                 alpha: float=0.5,
                 temperature: float=2.0,
                 n: int=1,
                 *args,
                 **kwargs):
        super().__init__(ce_alpha=alpha,
                         temperature=temperature,
                         *args,
                         **kwargs)
        self.n = n


class SequenceDistillationTrainer(Trainer):
    """
    ====== WIP ======
    Trainer class for distillation of sequential models.
    Should be used with `args=WordLevelDistillationTrainingArguments` or `args=SeqLevelDistillationTrainingArguments`.
    """
    def __init__(self,
                 teacher_model: PreTrainedModel,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.is_seq_level = isinstance(self.args, SeqLevelDistillationTrainingArguments)
    
    
    def compute_loss(self,
                     student_model: PreTrainedModel,
                     inputs,
                     return_outputs: bool=False):
        if not self.is_seq_level:  # If word-level distillation...
            return self._compute_word_level_loss(student_model, inputs, return_outputs)
        else:  # If sequence-level distillation...
            return self._compute_seq_level_loss(student_model, inputs, return_outputs)
    
    
    def _compute_word_level_loss(self,
                                 student_model: PreTrainedModel,
                                 inputs,
                                 return_outputs: bool=False):
        """
        ====== WIP ======
        """
        inputs = inputs.to(device)
        outputs_student = student_model(**inputs)
        
        # Extract cross-entropy loss and logits from student
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits
        
        # Extract logits from teacher
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits
        
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(  # type: ignore
            F.log_softmax(logits_student / self.args.temperature, dim=-1),  # type: ignore
            F.softmax(logits_teacher / self.args.temperature, dim=-1))  # type: ignore
        
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd  # type: ignore
        
        return (loss, outputs_student) if return_outputs else loss
    
    
    def _compute_seq_level_loss(self,
                                student_model: PreTrainedModel,
                                inputs,
                                return_outputs: bool=False):
        """
        ====== WIP ======
        Implementation of sequence-level distillation using the mode approximation.
        
        Reference: Kim et al., “Sequence-Level Knowledge Distillation.”, 2016, http://arxiv.org/abs/1606.07947.
        """
        raise NotImplementedError("Sequence-level distillation is not yet implemented.")

        inputs = inputs.to(device)
        # outputs_student = model(**inputs)
        
        # DEPRECATED: The gradient doesn't flow through the teacher model...
        # # Use N-beam search to get the most likely sequence:
        # mode_student = student_model.generate(**inputs, num_beams=self.args.n)  # type: ignore
        
        # Compute probability of sequence `mode_student` under teacher model:
        # Note that we need the gradient to flow through the teacher model. However, the
        # `generate` method does not allow this. Therefore, we used to following workaround:
        # https://github.com/huggingface/transformers/issues/15552#issuecomment-1033154753
        generate_with_grad = undecorated(student_model.generate)
        student_model.generate_with_grad = MethodType(generate_with_grad, student_model)  # type: ignore
        outputs_student = student_model.generate_with_grad(inputs, num_beams=self.args.n, output_scores=True, return_dict_in_generate=True)  # type: ignore
        
        # Note: `outputs_student.sequence` doesn't contain the gradients. Thus we need to compute the sequence
        #       using `argmax` and `outputs_student.scores`.
        # mode_student = torch.argmax(outputs_student.scores, ...)  # TOFILL
        
        # TODO: First, we have to check if the gradient flows from the student model's parameter and not just from `grad_fn=<CopySlices>`.
        
        # If ok, use the snippet below to compute the mode:
        # https://colab.research.google.com/drive/1Q8VAwCPB12ZzYH79nAuiiSSnVDkZ2-u7?usp=sharing#scrollTo=91HuEpJI-Z3X
        # with torch.no_grad():
        #   prob_seq_teacher = get_scores_for_labels(input=inputs, labels=mode_student, model=teacher_model, tokenizer=tokenizer)
        
        # Extract cross-entropy loss from student
        loss_ce = outputs_student.loss  # TODO: is this correct?
        
        # Soften probabilities and compute distillation loss
        # loss_kd = - self.args.temperature * torch.log(prob_seq_teacher)
        
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd  # type: ignore
        
        return (loss, outputs_student) if return_outputs else loss
