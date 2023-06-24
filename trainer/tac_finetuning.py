from typing import Optional, List

import torch

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedModel, WhisperForConditionalGeneration, WhisperProcessor
from transformers.feature_extraction_utils import BatchFeature
from utils.constants import GEN_MAX_LENGTH, LOSS_MASK_IDX

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
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        loss, output_student = super().compute_loss(model, inputs, return_outputs=True)
        
        for language_to_preserve in self.args.languages_to_preserve:
            tac_inputs = self.get_tac_inputs(inputs, language=language_to_preserve)
            model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language_to_preserve, task="transcribe")
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
        
        # Replace padding with correct token for correct loss computation:
        padded_mask = self.get_padded_mask_from_tensor(predicted_ids_tac)
        predicted_ids_tac = predicted_ids_tac.masked_fill(padded_mask.eq(1), LOSS_MASK_IDX)
        
        inputs_tac = inputs.copy()
        inputs_tac["labels"] = predicted_ids_tac
        
        return inputs_tac
    
    
    def get_padded_mask_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns the padded mask from a tensor of shape (batch_size, n_tokens).
        Used convention:
        - 1 for tokens that are padded
        - 0 otherwise.
        
        Example:
        - Input: tensor([[50257.,  50362.,     76.,    1694.,    627.,   50256.],
                         [50257.,  50362.,  13099.,   50256.,  50256.,   50256.]])
        - Output: tensor([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1]])
        """
        PAD_TOKEN_FROM_GENERATE = 50257  # different from the one used in the tokenizer
        assert tensor.ndim == 2, \
            f"The tensor must be 2D. Got {tensor.ndim} dimensions."
        
        indices = (tensor == PAD_TOKEN_FROM_GENERATE).long().argmax(dim=-1)
        padded_mask = torch.zeros_like(tensor, dtype=torch.long)
        for idx, row in zip(indices, padded_mask):
            row[idx+1:] = 1  # ignore the first EOT token of each row
        return padded_mask
    