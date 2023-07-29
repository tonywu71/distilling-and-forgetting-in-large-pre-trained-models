from typing import Optional, Callable
from toolz import valmap

import pandas as pd
from datasets import Dataset

from evaluation.string_edit_metrics import get_string_edit_metrics_ortho_and_norm
from normalization.whisper_normalization import get_whisper_normalizer


def eval_filter_criterion(data: pd.DataFrame | Dataset,
                          df_filter: Optional[pd.DataFrame] = None,
                          ds_filter: Optional[Callable] = None) -> None:
    if isinstance(data, pd.DataFrame):
        assert df_filter is not None, "`df_filter` must be provided if data is a DataFrame"
    elif isinstance(data, Dataset):
        assert ds_filter is not None, "`ds_filter` must be provided if data is a Dataset"

    n_rows_before = len(data)
    audio_length_before = sum(data["audio_length"]) / 60  # seconds to minutes
    metrics_before = get_string_edit_metrics_ortho_and_norm(references=data["text"], predictions=data["teacher_text"], norm_fn=get_whisper_normalizer("english"))
    metrics_before = valmap(lambda x: round(x, 2), metrics_before)

    if isinstance(data, pd.DataFrame):
        data = data[df_filter]
    elif isinstance(data, Dataset):
        data = data.filter(ds_filter)

    n_rows_after = len(data)
    audio_length_after = sum(data["audio_length"]) / 60  # seconds to minutes
    metrics_after = get_string_edit_metrics_ortho_and_norm(references=data["text"], predictions=data["teacher_text"], norm_fn=get_whisper_normalizer("english"))
    metrics_after = valmap(lambda x: round(x, 2), metrics_after)

    print(f"Number of rows before filtering: {n_rows_before}")
    print(f"Total audio length before filtering: {audio_length_before:.2f} minutes")
    print("String edit metrics before filtering:")
    print(pd.Series(metrics_before))

    print()

    print(f"Number of rows after filtering: {n_rows_after}")
    print(f"Total audio length after filtering: {audio_length_after:.2f} minutes")
    print("String edit metrics after filtering:")
    print(pd.Series(metrics_after))

    print()

    print(f"Number of rows removed: {n_rows_before - n_rows_after} ({100 * (n_rows_before - n_rows_after) / n_rows_before:.2f} %)")
    print(f"Total audio length removed: {audio_length_before - audio_length_after:.2f} minutes ({100 * (audio_length_before - audio_length_after) / audio_length_before:.2f} %)")

    print()

    print("String edit metrics difference:")
    print(pd.Series({key: round(metrics_before[key] - metrics_after[key], 2) for key in metrics_before.keys()}))

    print()

    print("Relative string edit metrics difference:")
    print(pd.Series({key: round((metrics_before[key] - metrics_after[key]) / metrics_before[key], 2) for key in metrics_before.keys()}))

    return
