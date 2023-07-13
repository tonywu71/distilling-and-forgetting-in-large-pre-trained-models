from typing import Dict, Any


def count_overlaps(result: Dict[str, Any]) -> int:
    counter = 0
    for segment in result["segments"]:
        for w1, w2 in zip(segment["words"], segment["words"][1:]):
            if w1["end"] > w2["start"]:
                counter += 1
    return counter


def check_words_within_delta(result, delta: float = 0.1, n_words: int = 5) -> bool:
    for segment in result['segments']:
        words = segment['words']
        for i in range(len(words) - n_words):
            if words[i + n_words]['start'] - words[i]['end'] <= delta:
                return True
    return False
