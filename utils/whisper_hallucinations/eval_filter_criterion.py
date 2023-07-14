from typing import Callable
from datasets import Dataset
from pprint import pprint
from toolz import valmap

from evaluation.string_edit_metrics import get_string_edit_metrics_ortho_and_norm
from normalization.whisper_normalization import get_whisper_normalizer


def eval_filter_criterion(ds: Dataset, filter_fn: Callable) -> None:
    n_rows_before = len(ds)
    audio_length_before = sum(ds["audio_length"]) / 60  # seconds to minutes
    metrics_before = get_string_edit_metrics_ortho_and_norm(references=ds["text"], predictions=ds["teacher_text"], norm_fn=get_whisper_normalizer("english"))
    metrics_before = valmap(lambda x: round(x, 2), metrics_before)

    ds_filered = ds.filter(filter_fn)

    n_rows_after = len(ds_filered)
    audio_length_after = sum(ds_filered["audio_length"]) / 60  # seconds to minutes
    metrics_after = get_string_edit_metrics_ortho_and_norm(references=ds_filered["text"], predictions=ds_filered["teacher_text"], norm_fn=get_whisper_normalizer("english"))
    metrics_after = valmap(lambda x: round(x, 2), metrics_after)

    print(f"Number of rows before filtering: {n_rows_before}")
    print(f"Total audio length before filtering: {audio_length_before:.2f} minutes")
    print("String edit metrics before filtering:")
    pprint(metrics_before)
    print()
    print(f"Number of rows after filtering: {n_rows_after}")
    print(f"Total audio length after filtering: {audio_length_after:.2f} minutes")
    print("String edit metrics after filtering:")
    pprint(metrics_after)
    print()
    print(f"Number of rows removed: {n_rows_before - n_rows_after} ({100 * (n_rows_before - n_rows_after) / n_rows_before:.2f} %)")
    print(f"Total audio length removed: {audio_length_before - audio_length_after:.2f} minutes ({100 * (audio_length_before - audio_length_after) / audio_length_before:.2f} %)")
    print("String edit metrics difference:")
    pprint({key: round(metrics_after[key] - metrics_before[key], 2) for key in metrics_before.keys()})
    print()
    print("Relative string edit metrics difference:")
    pprint({key: round((metrics_after[key] - metrics_before[key]) / metrics_before[key], 2) for key in metrics_before.keys()})

    return
