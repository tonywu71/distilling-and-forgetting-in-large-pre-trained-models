from datasets import Dataset


def sample_from_dataset(dataset, n_samples=10) -> Dataset:
    """Sample a small portion of the dataset."""
    samples = []
    for i, item in enumerate(dataset):
        samples.append(item)
        if i == (n_samples-1):
            break
    sample_dataset = Dataset.from_list(samples)
    return sample_dataset
