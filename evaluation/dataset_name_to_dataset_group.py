from functools import partial
from typing import Dict

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup

from dataloader.dataset_for_evaluation.esb_dataset import ESBDataset
from dataloader.dataset_for_evaluation.esb_dataset_with_ami_test import ESBDatasetWithAMITest
from dataloader.dataset_for_evaluation.esb_dataset_with_librispeech_test import ESBDatasetWithLibriSpeechTest
from dataloader.dataset_for_evaluation.fab_dataset import FABDataset
from dataloader.dataset_for_evaluation.mls_dataset import MLSDataset

from dataloader.dataset_for_evaluation.ami_test import AMITestSet
from dataloader.dataset_for_evaluation.librispeech_clean_test import LibriSpeechCleanTestSet


DATASET_NAME_TO_DATASET_GROUP: Dict[str, BaseDatasetGroup] = {
    "esb": partial(ESBDataset, load_diagnostic=True),
    "esb_librispeech": partial(ESBDatasetWithLibriSpeechTest, load_diagnostic=True),
    "esb_ami": partial(ESBDatasetWithAMITest, load_diagnostic=True),
    "fab": FABDataset,
    "mls": partial(MLSDataset, load_diagnostic=True),
    "ami_test": partial(AMITestSet),
    "librispeech_test": LibriSpeechCleanTestSet
}
