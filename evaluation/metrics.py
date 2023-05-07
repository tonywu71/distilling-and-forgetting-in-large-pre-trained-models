from typing import Dict
import evaluate
from transformers import WhisperProcessor

from utils.constants import PADDING_IDX


def compute_wer_fct(pred, processor: WhisperProcessor, normalize: bool=True) -> Dict[str, float]:
    """
    Compute the WER metric in percent for the given predictions and labels.
    Note: Setting `normalize` to `True` (default) will use the Whisper text normalizer.
    
    IMPORTANT: Due to a bug in the HuggingFace implementation of the Whisper, using
    `batch_decode` with `normalize=True` will always use the English normalizer even
    if the language is not English -> see https://github.com/huggingface/transformers/pull/20707
    For the moment, this is not a problem as we are always fine-tuning on English data, and
    the evaluation script doesn't use `batch_decode`.
    """
    
    wer_metric = evaluate.load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace the padding index with the pad token id to undo the step we applied
    # in the data collator to ignore padded tokens correctly in the loss:
    label_ids[label_ids==PADDING_IDX] = processor.tokenizer.pad_token_id  # type: ignore
    
    # Decode the predictions:
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore
    
    # Decode the labels:
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=normalize)  # type: ignore

    # Compute the WER in percent:
    wer = 100 * wer_metric.compute(references=label_str, predictions=pred_str)  # type: ignore

    return {"wer": wer}
