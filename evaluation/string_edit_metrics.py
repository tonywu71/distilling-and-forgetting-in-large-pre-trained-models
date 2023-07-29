from typing import Dict, List, Callable
import pandas as pd
from jiwer import compute_measures


def get_string_edit_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Return the string edit metrics (WER, substitutions, deletions, insertions) for the given predictions and references.
    
    Output:
        string_edit_metrics: dict with the following keys:
            - wer: word error rate,
            - sub: number of substitutions,
            - del: number of deletions,
            - ins: number of insertions.
    """
    incorrect = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    total = 0
    
    for prediction, reference in zip(predictions, references):
        if not reference:  # If the reference is empty...
            continue  # skip current iteration because the WER is not defined for empty references
        
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        substitutions += measures["substitutions"]
        deletions += measures["deletions"]
        insertions += measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    
    if total == 0:
        string_edit_metrics = {
            "wer": float('nan'),
            "sub": float('nan'),
            "del": float('nan'),
            "ins": float('nan'),
        }
    else:
        string_edit_metrics = {
            "wer": incorrect / total,
            "sub": substitutions / total,
            "del": deletions / total,
            "ins": insertions / total
        }
    
    return string_edit_metrics


def get_string_edit_metrics_ortho_and_norm(references: List[str],
                                           predictions: List[str],
                                           norm_fn: Callable[[str], str]) -> Dict[str, float]:
    dict_string_edit_metrics = {}
    
    # Compute the orthographic WER in percent and save it in the dictionary:
    string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=references, predictions=predictions))
    dict_string_edit_metrics["WER ortho (%)"] = string_edit_metrics["wer"]
    dict_string_edit_metrics["Sub ortho (%)"] = string_edit_metrics["sub"]
    dict_string_edit_metrics["Del ortho (%)"] = string_edit_metrics["del"]
    dict_string_edit_metrics["Ins ortho (%)"] = string_edit_metrics["ins"]

    # Get the normalized references and predictions (overwrites the previous lists to save memory):
    predictions = list(map(norm_fn, predictions))
    references = list(map(norm_fn, references))

    # Compute the normalized WER in percent and save it in the dictionary:
    string_edit_metrics = 100 * pd.Series(get_string_edit_metrics(references=references, predictions=predictions))
    dict_string_edit_metrics["WER (%)"] = string_edit_metrics["wer"]
    dict_string_edit_metrics["Sub (%)"] = string_edit_metrics["sub"]
    dict_string_edit_metrics["Del (%)"] = string_edit_metrics["del"]
    dict_string_edit_metrics["Ins (%)"] = string_edit_metrics["ins"]
    
    return dict_string_edit_metrics
