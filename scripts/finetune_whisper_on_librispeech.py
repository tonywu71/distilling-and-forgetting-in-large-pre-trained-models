import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from utils.initialize import initialize_env, print_envs
initialize_env()

from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pprint

from datasets import load_from_disk
from transformers import (WhisperForConditionalGeneration,
                          WhisperProcessor,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          EarlyStoppingCallback)

import wandb

from dataloader.dataloader import convert_dataset_dict_to_torch, load_dataset_dict, shuffle_dataset_dict
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.preprocessing import preprocess_dataset
from evaluation.metrics import compute_wer_fct
from utils.constants import GEN_MAX_LENGTH, WANDB_PROJECT
from utils.config import load_yaml_config


def main(config_filepath: str):
    """
    Fine-tune the Whisper model on the LibriSpeech dataset.
    """
    # --------------------   Load config   --------------------
    config = load_yaml_config(config_filepath)
    
    
    # -----------------------   W&B   -----------------------
    wandb.login()
    wandb.init(project=WANDB_PROJECT,
               job_type="whisper_finetuning",
               name=config.experiment_name,
               config=asdict(config))
    
    
    # ----------------------   Setup   ----------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("\n-----------------------\n")
    
    # Print environment variables:
    print("Environment variables:")
    print_envs()
    print("\n-----------------------\n")
    
    # Print config:
    print(f"Config loaded from `{config_filepath}`:")
    pprint(config)
    print("\n-----------------------\n")    
    
    
    # ----------------------   Main   ----------------------

    # Load processor (contains both tokenizer and feature extractor)
    processor = WhisperProcessor.from_pretrained(
        config.pretrained_model_name_or_path,
        language=config.lang_name,
        task="transcribe"
    )

    # Create the data collator that will be used to prepare the data for training:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    
    # Load the dataset and preprocess it:
    if Path(config.dataset_dir).exists():
        print(f"Previously saved dataset found at `{config.dataset_dir}`. Loading from disk...")
        dataset_dict = load_from_disk(config.dataset_dir)
    else:
        print(f"Dataset not found at `{config.dataset_dir}`. Extracting features from raw dataset...")
        dataset_dict = load_dataset_dict(dataset_name="librispeech")
        dataset_dict = preprocess_dataset(dataset_dict,
                                          tokenizer=processor.tokenizer,  # type: ignore
                                          feature_extractor=processor.feature_extractor,  # type: ignore
                                          augment=config.data_augmentation)
        dataset_dict = shuffle_dataset_dict(dataset_dict)
        dataset_dict = convert_dataset_dict_to_torch(dataset_dict)
        Path(config.dataset_dir).mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(config.dataset_dir)
        print(f"Dataset saved to `{config.dataset_dir}`. It will be loaded from disk next time.")
    
    
    # Initialize the model from a pretrained checkpoint:
    # Note: In Whisper, the decoder is conditioned on both the source and target sentences,
    #       and the decoder inputs are the concatenation of the target sentence and a special
    #       separator token. By default, the `forced_decoder_ids` attribute is set to a tensor
    #       containing the target sentence and the separator token. This tells the model to
    #       always generate the target sentence and the separator token before starting the
    #       decoding process.
    model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path).to(device)  # type: ignore
    model.config.forced_decoder_ids = None  # type: ignore
    model.config.suppress_tokens = []  # type: ignore
    model.config.use_cache = False  # type: ignore
    
    # Prepare training:
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.model_dir,
        do_eval=False,  # DEBUG
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-accumulation
        gradient_checkpointing=config.gradient_checkpointing,  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing
        fp16=True,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        generation_num_beams=config.generation_num_beams,
        logging_strategy="steps",
        logging_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,  # the lower the WER, the better
        report_to="wandb"  # type: ignore
    )
    
    compute_wer = partial(compute_wer_fct,
                          processor=processor,
                          normalize=True)
    
    callbacks = []
    if config.early_stopping_patience != -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,  # type: ignore
        train_dataset=dataset_dict["train"],  # type: ignore
        eval_dataset=dataset_dict["test"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_wer,  # type: ignore
        tokenizer=processor.tokenizer  # type: ignore
    )
    
    print("Starting training...")
    
    # Train the model:
    trainer.train()
    
    print("Training finished.")
    
    wandb.finish()
    return


if __name__ == "__main__":
    typer.run(main)
