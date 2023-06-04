from typing import Any, Dict
import inspect
from transformers import WhisperForConditionalGeneration


class WhisperForConditionalGenerationForDistillation(WhisperForConditionalGeneration):
    def forward(self, teacher_seqences=None, teacher_sequences_scores=None, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        
        # Remove "teacher_sequences" and "teacher_sequences_scores" from the model kwargs if exists:
        model_kwargs.pop("teacher_sequences", None)
        model_kwargs.pop("teacher_sequences_scores", None)
        
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
