from typing import Dict, List
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
