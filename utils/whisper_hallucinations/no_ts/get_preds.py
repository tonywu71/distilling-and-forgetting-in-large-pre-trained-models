from typing import List, Any, Tuple
from tqdm.auto import tqdm

from transformers import Pipeline
from datasets import Dataset
from dataloader.dataset_loader import gen_from_dataset

from utils.constants import DEFAULT_EVAL_NUM_BEAMS, GEN_MAX_LENGTH


def gen_preds_from_pipe(pipe: Pipeline, ds: Dataset) -> Tuple[List[str], List[Any]]:
    generate_kwargs = {"max_length": GEN_MAX_LENGTH,
                       "num_beams": DEFAULT_EVAL_NUM_BEAMS,
                       "language": "english",
                       "task": "transcribe"}
    # Create placeholders for the predictions and references:
    predictions = []
    references = []
    for out in tqdm(pipe(gen_from_dataset(ds),
                         batch_size=32,
                         generate_kwargs=generate_kwargs),
                    total=ds.num_rows):
        ref = out["reference"][0].lower()
        pred = out["text"].lower()
        references.append(ref)
        predictions.append(pred)
    return predictions, references
