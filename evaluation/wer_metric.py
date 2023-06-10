from typing import Dict

import pandas as pd
import torch
from transformers import WhisperProcessor, EvalPrediction

import wandb

from evaluation.string_edit_metrics import get_string_edit_metrics
from utils.constants import LOSS_MASK_IDX


def compute_wer_fct(pred: EvalPrediction,
                    processor: WhisperProcessor,
                    normalize: bool = True,
                    log_string_edit_metrics_on_wandb: bool = False) -> Dict[str, float]:
    """
    Compute the WER metric in percent for the given predictions and labels.
    Setting `normalize` to `True` (default) will use the Whisper text normalizer.
    
    IMPORTANT: Due to a bug in the HuggingFace implementation of the Whisper, using
    `batch_decode` with `normalize=True` will always use the English normalizer even
    if the language is not English -> see https://github.com/huggingface/transformers/pull/20707
    For the moment, this is not a problem as we are always fine-tuning on English data, and
    the evaluation script doesn't use `batch_decode`.
    """
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace the padding index with the pad token id to undo the step we applied
    # in the data collator to ignore padded tokens correctly in the loss:
    label_ids[label_ids==LOSS_MASK_IDX] = processor.tokenizer.pad_token_id  # type: ignore
    
    # Decode the predictions:
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore
    
    # Decode the labels:
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore

    # Compute the string edit metrics in percent:
    string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=label_str, predictions=pred_str))
    
    # Log the string edit metrics to wandb:
    if log_string_edit_metrics_on_wandb:
        wandb.log({"eval/sub_student_%": string_edit_metrics["sub"]})
        wandb.log({"eval/ins_student_%": string_edit_metrics["ins"]})
        wandb.log({"eval/del_student_%": string_edit_metrics["del"]})
    
    return {"wer": string_edit_metrics["wer"]}


def compute_wer_fct_distil(pred: EvalPrediction,
                           processor: WhisperProcessor,
                           normalize: bool=True,
                           log_string_edit_metrics_on_wandb: bool = False) -> Dict[str, float]:
    """
    Compute the WER metric in percent for the given predictions and labels for distillation.
    Setting `normalize` to `True` (default) will use the Whisper text normalizer.
    
    Note: This function cannot be used for sequence-level distillation as the 
    """
    
    # `pred` has the following attributes:
    # - predictions: Predictions of the model.
    # - label_ids: Targets to be matched.
    
    # For sequence`pred.predictions` is a 2-tuple:
    # - 1st element: the predictions of the student model -> (batch_size, seq_len, vocab_size) = (73, 92, 51865)
    # - 2nd element: the embeddings generated after the 2D convolution layers -> (73, 1500, 384)
    #                See `model.model.encoder.embed_positions (embed_positions): Embedding(1500, 384)
    
    pred_student = pred.predictions[0]
    pred_ids = torch.argmax(torch.Tensor(pred_student), dim=-1)  # type: ignore
    label_ids = pred.label_ids

    # Replace the padding index with the pad token id to undo the step we applied
    # in the data collator to ignore padded tokens correctly in the loss:
    label_ids[label_ids==LOSS_MASK_IDX] = processor.tokenizer.pad_token_id  # type: ignore
    
    # Decode the predictions:
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore
    
    # Decode the labels:
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore

    # Compute the string edit metrics in percent:
    string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=label_str, predictions=pred_str))
    
    # Log the string edit metrics to wandb:
    if log_string_edit_metrics_on_wandb:
        wandb.log({"eval/sub_student_%": string_edit_metrics["sub"]})
        wandb.log({"eval/ins_student_%": string_edit_metrics["ins"]})
        wandb.log({"eval/del_student_%": string_edit_metrics["del"]})
    
    return {"wer": string_edit_metrics["wer"]}
