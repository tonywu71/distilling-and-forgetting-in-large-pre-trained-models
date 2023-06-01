from typing import Any
from pathlib import Path
import pickle


def save_as_pickle(data: Any, savepath: str) -> None:
    """
    Save data as pickle file.
    If the parent directory does not exist, it will be created before saving.
    """
    if not Path(savepath).parent.exists():
        Path(savepath).parent.mkdir(parents=True)
    with open(savepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        return pickle.load(f)
