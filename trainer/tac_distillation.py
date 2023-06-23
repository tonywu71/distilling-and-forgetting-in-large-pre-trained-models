from typing import Optional, Literal, List

import torch

from transformers import PreTrainedModel, WhisperTokenizer

from models.model_utils import copy_model
from trainer.distillation import DistillationTrainingArguments, DistillationTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TACDistillationTrainingArguments(DistillationTrainingArguments):
    """
    Training arguments used for `TACDistillationTrainer`.
    """
    def __init__(self,
                 method_distil: Literal["word_level", "seq_level_k_best_uniform", "seq_level_k_best_ranked"],
                 languages_to_preserve: Optional[List[str]] = None,
                 method_tac: Optional[str] = None,
                 gamma_tac: float = 0.5,
                 alpha_ce: float = 0.5,
                 temperature: Optional[float] = None,
                 distillation_num_beams: Optional[int] = None,
                 beta_decay: Optional[float] = None,
                 *args,
                 **kwargs):
        super().__init__(method=method_distil,
                         alpha_ce=alpha_ce,
                         temperature=temperature,
                         distillation_num_beams=distillation_num_beams,
                         beta_decay=beta_decay,
                         *args,
                         **kwargs)
        self.gamma_tac = gamma_tac
        self.languages_to_preserve = languages_to_preserve if languages_to_preserve is not None else []
        
        if method_tac is None:
            print(f"WARNING: `method_tac` is not set. Using `method_tac={method_distil}`.")
            self.method_tac = method_distil
        else:
            self.method_tac = method_tac


class TACDistillationTrainer(DistillationTrainer):
    """
    Trainer class for distillation with Task Alignment Consolidation (TAC).
    Should be used with `args=TACDistillationTrainingArguments`.
    """
    def __init__(self,
                 student_tokenizer: Optional[WhisperTokenizer] = None,
                 teacher_model: Optional[PreTrainedModel] = None,
                 *args,
                 **kwargs):
        super().__init__(student_tokenizer=student_tokenizer,
                         teacher_model=teacher_model,
                         *args,
                         **kwargs)
        
        assert self.args.method_tac == "word_level", \
            "Task Alignment Consolidation (TAC) is only supported for `word_level` distillation for now."
        
        # TODO: Add support for `seq_level_k_best_uniform` and `seq_level_k_best_ranked` distillation
        #       This will require to run k-beam search on the orginial student model as well... thus
        #       we will need run smart caching one more time in the distil script.
        
        self.original_student_model = copy_model(self.student_model)
        
        # Freeze the parameters of self.original_student_model:
        for param in self.original_student_model.parameters():
            param.requires_grad = False
    
    
    def compute_loss(self,
                     student_model: PreTrainedModel,
                     inputs,
                     return_outputs: bool = False):
        """
        Override the `compute_loss` method from `DistillationTrainer`.
        Computes the loss according to the distillation method specified in `self.args.method` and
        the TAC method specified in `self.args.method_tac`.
        """
        loss, output_student = self.METHOD_TO_LOSS_FCT[self.args.method](student_model=student_model,
                                                                         inputs=inputs,
                                                                         teacher_model=self.teacher_model,
                                                                         language="english")
        
        if self.args.tac_regularization:
            for language_to_preserve in self.args.languages_to_preserve:
                loss_other_task, _ = self.METHOD_TO_LOSS_FCT[self.args.method_tac](student_model=student_model,
                                                                                   inputs=inputs,
                                                                                   teacher_model=self.original_student_model,
                                                                                   language=language_to_preserve)
                loss += self.args.tac_gamma * loss_other_task / len(self.args.languages_to_preserve)
        
        return (loss, output_student) if return_outputs else loss