import pytest

from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor

from dataloader.dataloader_custom import rename_label_col
from dataloader.preprocessing import preprocess_dataset
from utils.config import Config


DEFAULT_PRETRAINED_MODEL_NAME_OR_PATH = "openai/whisper-tiny.en"


@pytest.fixture
def config() -> Config:
    config = Config(
        experiment_name="debug_0",
        lang_name="english",
        lang_id="en",
        pretrained_model_name_or_path="openai/whisper-tiny.en",
        model_dir="./checkpoints",
        batch_size=128,
        data_augmentation=False,
        dataset_dir="",
        optim="adamw_hf",
        learning_rate=1.5e-3,
        warmup_steps=256,
        num_train_epochs=1
    )
    return config


@pytest.fixture
def dataset_dict() -> DatasetDict:
    dataset_dict = {}
    dataset_dict["validation"] = load_dataset("hf-internal-testing/librispeech_asr_dummy", name="clean", split="validation")
    dataset_dict = DatasetDict(dataset_dict)
    
    dataset_dict = rename_label_col(dataset_dict, old_label_col="text")
    
    return dataset_dict


@pytest.fixture
def processor(config: Config) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(
        config.pretrained_model_name_or_path,
        language=config.lang_name,
        task="transcribe"
    )
    return processor


def test_preprocess_dataset(dataset_dict: DatasetDict, config: Config, processor: WhisperProcessor):
    # Preprocess the dataset:
    dataset_dict = preprocess_dataset(dataset_dict,
                                      tokenizer=processor.tokenizer,  # type: ignore
                                      feature_extractor=processor.feature_extractor,  # type: ignore
                                      augment=config.data_augmentation)
    
    # Assert validation set is not empty:
    test_sample = next(iter(dataset_dict["validation"]))
    assert test_sample is not None, "validation split is empty"
    
    assert "input_features" in dataset_dict["validation"].features, "input_features not found"
