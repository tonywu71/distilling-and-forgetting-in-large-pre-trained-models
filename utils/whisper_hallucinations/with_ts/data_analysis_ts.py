from typing import Dict, Any

def count_overlaps(result: Dict[str, Any]) -> int:
    counter = 0
    for segment in result["segments"]:
        for w1, w2 in zip(segment["words"], segment["words"][1:]):
            if w1["end"] > w2["start"]:
                counter += 1
    return counter
