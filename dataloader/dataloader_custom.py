from datasets import load_dataset


def load_librispeech(**kwargs) -> dict:
    """Load the LibriSpeech dataset."""
    dataset_dict = {}
    dataset_dict["train"] = load_dataset("librispeech_asr", name="clean", split="train", streaming=True)
    dataset_dict["test"] = load_dataset("librispeech_asr", name="clean", split="test", streaming=True)
    return dataset_dict
