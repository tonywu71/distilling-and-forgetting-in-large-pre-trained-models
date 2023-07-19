import re


def remove_punctuation(text):
    """
    Removes punctuation from a string, but keeps apostrophes.
    """
    return re.sub(r'[^\w\s\']', '', text)


def remove_casing_and_punctuation(text: str) -> str:
    """
    Removes casing and punctuation from a string, but keeps apostrophes.
    """
    return remove_punctuation(text).lower()
