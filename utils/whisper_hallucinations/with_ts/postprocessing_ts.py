from typing import Any, Dict
from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from utils.constants import GEN_MAX_LENGTH


def _truncate_tokens(result: Dict[str, Any], max_length: int = GEN_MAX_LENGTH) -> Dict[str, Any]:
    """
    Important: Only the tokens are truncated, not the words!
    """
    tokens = []
    for segment in result['segments']:
        tokens.extend(segment['tokens'])
    result['tokens'] = tokens[:max_length]  # NOTE: new key
    return result


def truncate_results(results: Dict[str, Any],
                     tokenizer: WhisperTokenizer | WhisperTokenizerFast,
                     max_length: int = GEN_MAX_LENGTH) -> Dict[str, Any]:
    results = [_truncate_tokens(x, max_length=max_length) for x in results]

    truncated_texts = tokenizer.batch_decode([result["tokens"] for result in results])
    for result, new_text in zip(results, truncated_texts):
        result["text"] = new_text
    
    return results
