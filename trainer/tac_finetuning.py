from typing import Optional, List

import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedModel
from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from transformers.feature_extraction_utils import BatchFeature

from trainer.trainer_utils import get_padded_mask_from_tensor
from utils.constants import GEN_MAX_LENGTH, LOSS_MASK_IDX


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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Creating a copy of the original model...")
        self.original_model = WhisperForConditionalGeneration.from_pretrained(self.model.config._name_or_path).to(device)
        
        self.original_model.config.suppress_tokens = []
        
        # Freeze the copy of the original model:
        for param in self.original_model.parameters():
            param.requires_grad = False
        self.original_model._requires_grad = False
    
    
    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs: BatchFeature,
                     return_outputs: bool = False):
        """
        Override the `compute_loss` method from `Seq2SeqTrainer`.
        """
        loss, output_student = super().compute_loss(model, inputs, return_outputs=True)
        
        for language_to_preserve in self.args.languages_to_preserve:
            tac_inputs = self.get_tac_inputs(inputs, language=language_to_preserve)
            loss_other_task = super().compute_loss(model, tac_inputs, return_outputs=False)
            loss += self.args.gamma_tac * loss_other_task / len(self.args.languages_to_preserve)
        
        # Reset the forced decoder ids to the original value:
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        
        return (loss, output_student) if return_outputs else loss

    
    def get_tac_inputs(self, inputs: BatchFeature, language: str) -> BatchFeature:
        """
        Get the TAC inputs. They are built by replacing the labels from `inputs` with the predicted labels from the original model
        for the current language.
        """
        self.original_model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
        
        predicted_ids_tac = self.original_model.generate(inputs["input_features"],  # greedy decoding
                                                         max_length=GEN_MAX_LENGTH)
        
        # TODO: Check if this is correct...
        
        # Replace padding with correct token for correct loss computation:
        padded_mask = get_padded_mask_from_tensor(predicted_ids_tac)
        predicted_ids_tac = predicted_ids_tac.masked_fill(padded_mask.eq(1), LOSS_MASK_IDX)
        
        inputs_tac = inputs.copy()
        inputs_tac["labels"] = predicted_ids_tac
        
        return inputs_tac
