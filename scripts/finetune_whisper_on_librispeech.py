import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from dataclasses import asdict

from transformers import (WhisperForConditionalGeneration,
                          WhisperProcessor,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          WhisperTokenizer,
                          WhisperFeatureExtractor)

import wandb


from dataloader.dataloader import convert_dataset_dict_to_torch, load_dataset_dict, shuffle_dataset_dict
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.preprocessing import preprocess_dataset
from models.callbacks import ShuffleCallback
from evaluation.metrics import compute_wer
from utils.config import parse_yaml_config


def main(config_filepath: str):
    """
    Fine-tune the Whisper model on the LibriSpeech dataset.
    """
    config = parse_yaml_config(config_filepath)

    # -----------------------   W&B   -----------------------
    wandb.login()

    # initialize tracking the experiment
    wandb.init(project="whisper_finetuning",
               job_type="fine-tuning",
               name=config.experiment_name,
               config=asdict(config))

    
    # ----------------------   Main   ----------------------

    # Load feature extractor, tokenizer and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.pretrained_model_name_or_path
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        language=config.lang_name,
        task="transcribe"
    )

    processor = WhisperProcessor.from_pretrained(
        config.pretrained_model_name_or_path,
        task="transcribe"
    )

    # Create the data collator that will be used to prepare the data for training:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    
    # Load the dataset and preprocess it:
    dataset_dict = load_dataset_dict(dataset_name="librispeech")
    
    dataset_dict = preprocess_dataset(dataset_dict, feature_extractor=feature_extractor)
    dataset_dict = shuffle_dataset_dict(dataset_dict)
    dataset_dict = convert_dataset_dict_to_torch(dataset_dict)
    
    
    # Initialize the model from a pretrained checkpoint:
    model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path)
    model.config.forced_decoder_ids = None  # type: ignore
    model.config.suppress_tokens = []  # type: ignore
    model.config.use_cache = False  # type: ignore
    
    # Prepare training:
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.model_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,  # the lower the WER, the better
        # remove_unused_columns=False, 
    )
    
    callbacks = [ShuffleCallback()]

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,  # type: ignore
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        compute_metrics=compute_wer,  # type: ignore
        tokenizer=tokenizer,
        callbacks=callbacks  # type: ignore
    )
    
    # Train the model:
    trainer.train()
    
    
    wandb.finish()
    return


if __name__ == "__main__":
    typer.run(main)
