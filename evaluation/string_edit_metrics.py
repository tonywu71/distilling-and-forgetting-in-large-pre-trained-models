from re import sub
from typing import Dict
from jiwer import compute_measures


def get_string_edit_metrics(predictions: str, references: str) -> Dict[str, float]:
    """
    Return the string edit metrics (WER, substitutions, deletions, insertions) for the given predictions and references.
    
    Output:
        string_edit_metrics: dict with the following keys:
            - wer: word error rate,
            - sub: number of substitutions,
            - def: number of deletions,
            - ins: number of insertions.
    """
    incorrect = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    total = 0
    
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        substitutions += measures["substitutions"]
        deletions += measures["deletions"]
        insertions += measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    
    string_edit_metrics = {
        "wer": incorrect / total,
        "sub": substitutions / total,
        "del": deletions / total,
        "ins": insertions / total
    }
    
    return string_edit_metrics
