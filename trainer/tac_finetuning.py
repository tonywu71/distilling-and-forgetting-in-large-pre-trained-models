from typing import Optional, List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from transformers.feature_extraction_utils import BatchFeature
from optimum.bettertransformer import BetterTransformer


from trainer.trainer_utils import get_language_special_token, get_padded_mask_from_tensor
from utils.constants import GEN_MAX_LENGTH, LOSS_MASK_IDX


class TACFinetuningTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments for fine-tuning with Task Alignment Consolidation (TAC).
    Should be used with `TACFinetuningTrainer`.
    """
    def __init__(self,
                 languages_to_preserve: Optional[List[str]] = None,
                 gamma_tac: float = 0.5,
                 task_tac: str = "transcribe",
                 use_kl: bool = False,
                 temperature: float = 1.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_tac = gamma_tac
        self.languages_to_preserve = languages_to_preserve if languages_to_preserve is not None else []
        self.task_tac = task_tac
        self.use_kl = use_kl
        self.temperature = temperature


class TACFinetuningTrainer(Seq2SeqTrainer):
    """
    Trainer class for fine-tuning with Task Alignment Consolidation (TAC).
    Should be used with `args=TACFinetuningTrainingArguments`.
    """
    def __init__(self,
                 args: TACFinetuningTrainingArguments,
                 processor: WhisperProcessor,
                 **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.processor = processor
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16  # see https://huggingface.co/learn/audio-course/chapter5/evaluation?fw=pt
        elif torch.backends.mps.is_available():  # for Apple Silicon
            self.device = torch.device('mps')
            self.torch_dtype = torch.float32  # float16 not supported by MPS
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
        print("Creating a copy of the original model...")
        self.original_model = WhisperForConditionalGeneration.from_pretrained(self.model.config._name_or_path).to(self.device).to(self.torch_dtype)
        
        if torch.cuda.is_available():
            print("CUDA is available. Transforming the original model to use the BetterTransformer...")
            self.original_model_model = BetterTransformer.transform(self.original_model)

        # Freeze the copy of the original model:
        for param in self.original_model.parameters():
            param.requires_grad = False
        self.original_model._requires_grad = False

        self.original_model.generate = partial(self.original_model.generate, task=self.args.task_tac, use_cache=True)
    
    
    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs: BatchFeature,
                     return_outputs: bool = False):
        """
        Override the `compute_loss` method from `Seq2SeqTrainer`.
        """

        # Compute the default cross-entropy loss for the target task:
        loss, outputs_target = super().compute_loss(model, inputs, return_outputs=True)
        
        # Compute the cross-entropy loss for the other tasks:
        for language_to_preserve in self.args.languages_to_preserve:
            if self.args.use_kl:
                loss_other_task = self.compute_tac_loss_with_kd(model, inputs, language_to_preserve)
            else:
                tac_inputs = self.get_tac_inputs(inputs, language=language_to_preserve)
                loss_other_task = super().compute_loss(model, tac_inputs, return_outputs=False)
            breakpoint()
            loss += self.args.gamma_tac * loss_other_task / len(self.args.languages_to_preserve)
        return (loss, outputs_target) if return_outputs else loss

    
    def get_tac_inputs(self, inputs: BatchFeature, language: str) -> BatchFeature:
        """
        Get the TAC inputs. They are built by replacing the labels from `inputs` with the predicted labels from the original model
        for the current language.
        """
        predicted_ids_tac = self.original_model.generate(inputs["input_features"].to(self.device).to(self.torch_dtype),
                                                         language=language, max_length=GEN_MAX_LENGTH)
        
        # NOTE: `generate` returns a tensor with the SOT token at the beginning. However,
        #       `compute_loss` expects the tensor to start with the first token of the target sequence.
        #       Therefore, we have to remove the first token from the predicted tensor.

        # Remove the first token (SOT) from the predicted tensor:
        predicted_ids_tac = predicted_ids_tac[:, 1:]
        
        # Replace padding with correct token for correct loss computation:
        padded_mask = get_padded_mask_from_tensor(predicted_ids_tac)
        predicted_ids_tac = predicted_ids_tac.masked_fill(padded_mask.eq(1), LOSS_MASK_IDX)
        
        # Override the labels from `inputs` with the predicted labels from the original model:
        inputs_tac = inputs.copy()
        inputs_tac["labels"] = predicted_ids_tac
        
        return inputs_tac


    def compute_tac_loss_with_kd(self, model, inputs, language_to_preserve: str):
        """
        Compute the TAC loss with knowledge distillation.
        """
        inputs_tac = inputs.copy().to(self.device).to(self.torch_dtype)

        # Replace language token with correct one:
        new_lang_token = get_language_special_token(language_to_preserve)
        lang_token_tensor = torch.LongTensor([[new_lang_token]]).expand(inputs_tac["labels"].shape[0], 1).to(self.device)
        inputs_tac["labels"] = torch.cat([lang_token_tensor, inputs_tac["labels"][:, 1:]], dim=1)

        # Forward pass through original model (teacher-forced):
        # FIXME: Right now, we are using teacher-forcing on the wrong labels (English).
        #        An easy fix would be to first generate the new labels using `self.original_model.generate`
        #        and to use them as the `decoder_input_ids` argument for both models.
        with torch.no_grad():
            original_logits = self.original_model.forward(**inputs_tac).logits

        # Forward pass through currently trained model (teacher-forced):
        target_logits = model.forward(**inputs_tac).logits
        
        # Initialize KL-divergence loss:
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Important:
        # - `KLDivLoss` argument order is the opposite of the one for the KL(·||·) mathematical notation
        # - `KLDivLoss` expects log-probabilities for `input` to avoid underflow issues
        
        # Soften probabilities and compute distillation loss:
        # NOTE: `input` should be log-probabilities according to the documentation of `KLDivLoss`.
        loss_kd = self.args.temperature ** 2 * kl_div_loss(
            input=F.log_softmax(original_logits / self.args.temperature, dim=-1),
            target=F.softmax(target_logits / self.args.temperature, dim=-1))  # (1,)

        return loss_kd
