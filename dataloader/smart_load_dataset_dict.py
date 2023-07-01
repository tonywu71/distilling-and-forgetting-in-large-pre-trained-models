import os
import shutil
from pathlib import Path

from transformers.models.whisper import WhisperProcessor
from datasets import load_from_disk, DatasetDict, Dataset

from dataloader.dataloader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import preprocess_dataset
from dataloader.utils import get_fast_tokenizer_from_tokenizer
from utils.finetune_config import FinetuneConfig
from utils.distil_config import DistilConfig


def smart_load_dataset_dict(config: FinetuneConfig | DistilConfig,
                            processor: WhisperProcessor,
                            fast_tokenizer: bool = True) -> DatasetDict | Dataset:
    # Sanity check:
    processed_datasets_dir = Path(os.environ["PREPROCESSED_DATASETS_DIR"])
    if not processed_datasets_dir.exists():
        print(f"Preprocessed datasets directory `{os.environ['PREPROCESSED_DATASETS_DIR']}` not found. A new directory will be created.")
        processed_datasets_dir.mkdir(parents=True)
    
    
    # Get the path to the preprocessed dataset:
    tokenizer_name = "multilingual_tokenizer" if config.is_tokenizer_multilingual else "en_tokenizer"
    dataset_dir = str(Path(os.environ["PREPROCESSED_DATASETS_DIR"]) / config.dataset_name / tokenizer_name)
    
    
    # Load the preprocessed dataset if it exists and if `force_reprocess_dataset` is set to `False`:
    if not config.force_reprocess_dataset and Path(dataset_dir).exists():
        print(f"Previously preprocessed dataset found at `{dataset_dir}`. Loading from disk...")
        dataset_dict = load_from_disk(dataset_dir)
    
    # Otherwise, preprocess the dataset from scratch:
    else:
        if config.force_reprocess_dataset:
            print(f"`force_reprocess_dataset` was set to `True` in the config file.")
            if Path(dataset_dir).exists():
                print(f"Deleting previously preprocessed dataset at `{dataset_dir}`...")
                shutil.rmtree(dataset_dir) 
        else:
            print(f"Preprocessed dataset not found at `{dataset_dir}`.")
        
        
        print("Preprocessing from scratch...")
        
        print(f"Loading raw dataset `{config.dataset_name}` from Huggingface...")
        dataset_dict = load_dataset_dict(dataset_name=config.dataset_name)
        
        print(f"Preprocessing dataset `{config.dataset_name}`...")
        tokenizer = get_fast_tokenizer_from_tokenizer(processor.tokenizer) if fast_tokenizer else processor.tokenizer
        dataset_dict = preprocess_dataset(dataset_dict,  # type: ignore
                                          tokenizer=tokenizer,
                                          feature_extractor=processor.feature_extractor,  # type: ignore
                                          augment=config.data_augmentation)
        
        # NOTE: The pytorch tensor conversation will be done in the DataCollator.
        
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(dataset_dir)
        print(f"Preprocessed dataset saved to `{dataset_dir}`. It will be loaded from disk next time.")
    
    return dataset_dict
