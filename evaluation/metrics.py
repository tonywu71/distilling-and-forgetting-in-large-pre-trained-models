from typing import Dict
import evaluate
from transformers import WhisperProcessor

from utils.constants import PADDING_IDX


def compute_wer_fct(pred, processor: WhisperProcessor, normalize: bool=True) -> Dict[str, float]:
    """
    Compute the WER metric in percent for the given predictions and labels.
    Note: Setting `normalize` to `True` (default) will use the Whisper text normalizer.
    """
    
    metric = evaluate.load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace the padding index with the pad token id so that the metric can be computed correctly:
    label_ids[label_ids == PADDING_IDX] = processor.tokenizer.pad_token_id  # type: ignore
    
    # Decode the predictions:
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore
    
    # Decode the labels:
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore

    # Compute the WER in percent:
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)  # type: ignore

    return {"wer": wer}
