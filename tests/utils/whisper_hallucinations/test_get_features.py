from utils.whisper_hallucinations.get_features import max_contiguous_ngrams


def test_max_contiguous_ngrams():
    text = "So, what we will do is, we will do is, we will do is, we will do is, we will do is,"
    assert max_contiguous_ngrams(text, 1) == 1
    assert max_contiguous_ngrams(text, 2) == 1
    assert max_contiguous_ngrams(text, 3) == 1
    assert max_contiguous_ngrams(text, 4) == 5
    assert max_contiguous_ngrams(text, 5) == 1
