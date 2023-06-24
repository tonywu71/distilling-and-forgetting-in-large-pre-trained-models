from typing import Optional, List

import torch

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedModel, WhisperForConditionalGeneration, WhisperProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TACFinetuningTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments used for `TACFinetuningTrainer`.
    """
    def __init__(self,
                 languages_to_preserve: Optional[List[str]] = None,
                 gamma_tac: float = 0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_tac = gamma_tac
        self.languages_to_preserve = languages_to_preserve if languages_to_preserve is not None else []


class TACFinetuningTrainer(Seq2SeqTrainer):
    """
    Trainer class for distillation with Task Alignment Consolidation (TAC).
    Should be used with `args=TACFinetuningTrainingArguments`.
    """
    def __init__(self,
                 processor: WhisperProcessor,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.processor = processor
        
        print("Creating a copy of the original model...")
        self.original_model = WhisperForConditionalGeneration.from_pretrained(self.model.config._name_or_path).to(device)  # type: ignore
        
        # Freeze the copy of the original model:
        for param in self.original_model.parameters():
            param.requires_grad = False
        self.original_model._requires_grad = False
    
    
    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs,
                     return_outputs: bool = False):
        """
        Override the `compute_loss` method from `Seq2SeqTrainer`.
        """
        
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        loss, output_student = super().compute_loss(model, inputs, return_outputs=True)
        
        for language_to_preserve in self.args.languages_to_preserve:
            model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language_to_preserve, task="transcribe")
            loss_other_task = super().compute_loss(model, inputs, return_outputs=False)
            loss += self.args.gamma_tac * loss_other_task / len(self.args.languages_to_preserve)
        
        # Reset the forced decoder ids to the original value:
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        
        return (loss, output_student) if return_outputs else loss
