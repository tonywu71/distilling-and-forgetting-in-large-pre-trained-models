from typing import List, Optional
from IPython.display import Audio, display
from datasets import Dataset


def listen_to_audio(ds: Dataset, list_idx: List[int], pred_col: Optional[str]) -> None:
    assert "audio" in ds.features, "The dataset must have an 'audio' column."
    ds_sampled = ds.select(list_idx)
    for idx, sample in zip(list_idx, ds_sampled):
        print(f"Idx: {idx}")
        print(f"Reference: {sample['text']}")
        if pred_col:
            print(f"Prediction: {sample[pred_col]}")
        display(Audio(data=sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
        print()
    return
