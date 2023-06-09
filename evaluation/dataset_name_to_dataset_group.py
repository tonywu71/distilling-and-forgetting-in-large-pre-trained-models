from functools import partial
from typing import Dict

from dataloader.datasets.base_dataset_group import BaseDatasetGroup

from dataloader.datasets.esb_dataset import ESBDataset
from dataloader.datasets.esb_dataset_with_ami_test import ESBDatasetWithAMITest
from dataloader.datasets.esb_dataset_with_librispeech_test import ESBDatasetWithLibriSpeechTest
from dataloader.datasets.fab_dataset import FABDataset
from dataloader.datasets.mls_dataset import MLSDataset

from dataloader.datasets.ami_test import AMITestSet
from dataloader.datasets.librispeech_clean_test import LibriSpeechCleanTestSet


DATASET_NAME_TO_DATASET_GROUP: Dict[str, BaseDatasetGroup] = {
    "esb": partial(ESBDataset, load_diagnostic=True),
    "esb_librispeech": partial(ESBDatasetWithLibriSpeechTest, load_diagnostic=True),
    "esb_ami": partial(ESBDatasetWithAMITest, load_diagnostic=True),
    "fab": FABDataset,
    "mls": partial(MLSDataset, load_diagnostic=True),
    "ami_test": partial(AMITestSet, is_ami_10h=True),
    "librispeech_test": LibriSpeechCleanTestSet
}
