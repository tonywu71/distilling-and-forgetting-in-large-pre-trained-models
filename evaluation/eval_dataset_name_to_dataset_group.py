from functools import partial
from typing import Dict

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from dataloader.dataset_for_evaluation.librispeech_clean_test import LibriSpeechCleanTestSet
from dataloader.dataset_for_evaluation.librispeech_dummy_dataset import LibriSpeechDummyDataset
from dataloader.dataset_for_evaluation.ami_validation import AMIValidationSet
from dataloader.dataset_for_evaluation.ami_test import AMITestSet
from dataloader.dataset_for_evaluation.ami_eval import AMIEvalSet
from dataloader.dataset_for_evaluation.esb_diagnostic_dataset import ESBDiagnosticDataset
from dataloader.dataset_for_evaluation.esb_diagnostic_custom_dataset import ESBDiagnosticCustomDataset
from dataloader.dataset_for_evaluation.fab_dataset import FABDataset
from dataloader.dataset_for_evaluation.mls_dataset import MLSDataset
from dataloader.dataset_for_evaluation.mls_diagnostic_dataset_custom import MLSDiagnosticDatasetCustom


EVAL_DATASET_NAME_TO_DATASET_GROUP: Dict[str, BaseDatasetGroup] = {
    "librispeech_clean": LibriSpeechCleanTestSet,
    "librispeech_dummy": LibriSpeechDummyDataset,
    "ami_validation": AMIValidationSet,
    "ami_validation_10h": partial(AMIValidationSet, is_ami_10h=True),
    "ami": AMITestSet,
    "ami_10h": partial(AMITestSet, is_ami_10h=True),
    "ami_eval": AMIEvalSet,
    "esb_diagnostic": ESBDiagnosticDataset,
    "esbdc": ESBDiagnosticCustomDataset,
    "mls": MLSDataset,
    "mlsdc": MLSDiagnosticDatasetCustom,
    "fab": FABDataset
}
