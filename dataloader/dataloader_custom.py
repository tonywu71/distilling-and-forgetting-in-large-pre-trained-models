from datasets import load_dataset


def load_librispeech(**kwargs) -> dict:
    """Load the LibriSpeech dataset."""
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("librispeech_asr", name="clean", split="train.100")
    dataset_dict["test"] = load_dataset("librispeech_asr", name="clean", split="test")
    return dataset_dict
