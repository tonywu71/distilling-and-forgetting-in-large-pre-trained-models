from datasets import DatasetDict


def rename_label_col(dataset_dict: DatasetDict,
                     old_label_col: str,
                     new_label_col: str) -> DatasetDict:
    """
    Rename the label column in dataset_dict.
    """
    
    if old_label_col == new_label_col:  # otherwise, we will get an error
        return dataset_dict
    
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].rename_column(old_label_col, new_label_col)
    
    return dataset_dict
