from typing import Dict

from transformers import WhisperProcessor, EvalPrediction

from toolz import dicttoolz

from evaluation.string_edit_metrics import get_string_edit_metrics
from utils.constants import LOSS_MASK_IDX


def compute_string_edit_metrics_fct(pred: EvalPrediction,
                                    processor: WhisperProcessor,
                                    normalize: bool = True) -> Dict[str, float]:
    """
    Compute the string edit metrics (WER, substitutions, deletions, insertions) in percent
    for the given predictions and labels.
    
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
    string_edit_metrics = get_string_edit_metrics(references=label_str, predictions=pred_str)
    string_edit_metrics = dicttoolz.valmap(lambda x: x * 100, string_edit_metrics)
    
    return string_edit_metrics  # keys: (wer, sub, del, ins)
