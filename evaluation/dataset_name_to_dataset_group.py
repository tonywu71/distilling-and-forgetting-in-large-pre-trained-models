from functools import partial
from typing import Dict

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup

from dataloader.dataset_for_evaluation.esb_diagnostic_dataset import ESBDiagnosticDataset
from dataloader.dataset_for_evaluation.esb_diagnostic_custom_dataset import ESBDiagnosticCustomDataset
from dataloader.dataset_for_evaluation.fab_dataset import FABDataset
from dataloader.dataset_for_evaluation.mls_dataset import MLSDataset
from dataloader.dataset_for_evaluation.ami_test import AMITestSet
from dataloader.dataset_for_evaluation.librispeech_clean_test import LibriSpeechCleanTestSet


DATASET_NAME_TO_DATASET_GROUP: Dict[str, BaseDatasetGroup] = {
    "librispeech_clean_test": LibriSpeechCleanTestSet,
    "ami_test": AMITestSet,
    "ami_10h_test": partial(AMITestSet, is_ami_10h=True),
    "esb_diagnostic": ESBDiagnosticDataset,
    "esb_diagnostic_custom": ESBDiagnosticCustomDataset,
    "mls": partial(MLSDataset, load_diagnostic=True),
    "fab": FABDataset,
}
