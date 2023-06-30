import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env, print_envs
initialize_env()

from typing import List
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pprint

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.whisper import (WhisperForConditionalGeneration,
                                         WhisperTokenizerFast,
                                         WhisperFeatureExtractor,
                                         WhisperProcessor)
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback

import wandb

from callbacks.eval_first_step_callback import EvalFirstStepCallback
from callbacks.finetune_callback import WandbFinetuneCallback
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataloader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import preprocess_dataset
from dataloader.smart_load_dataset_dict import smart_load_dataset_dict
from evaluation.wer_metric import compute_string_edit_metrics_fct
from models.whisper_zero_cross_attention import WhisperForConditionalGenerationZeroCrossAttention
from trainer.tac_finetuning import TACFinetuningTrainer, TACFinetuningTrainingArguments
from utils.constants import GEN_MAX_LENGTH
from utils.file_io import fix_model_dir_conflicts
from utils.finetune_config import FinetuneConfig
from utils.tac_finetune_config import TACFinetuneConfig


def main(config_filepath: str,
         tac: bool = typer.Option(False, help="Whether to use Task Alignment Consolidation or not. " + \
                                              "Flag should be used if only if the config file is for TAC distillation. " + \
                                              "TAC is not compatible with zero-shot.")):
    """
    Fine-tune the Whisper model on the LibriSpeech dataset.
    """
    # --------------------   Load config   --------------------
    if tac:
        config = TACFinetuneConfig.from_yaml(config_filepath)
    else:
        config = FinetuneConfig.from_yaml(config_filepath)
    
    # If a previous run has its checkpoints saved in the same directory,
    # add a timestamp to the model directory. This is to avoid overwriting
    # previous models. Note that `config` is modified in-place.
    fix_model_dir_conflicts(config)
    
    # Prepare tags for W&B:
    list_tags = [config.dataset_name]
    if tac:
        list_tags.append("tac")
    
    # -----------------------   W&B   -----------------------
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_TRAINING"],
               job_type="finetuning",
               tags=list_tags,
               name=config.experiment_name,
               config=asdict(config))
    
    
    # ----------------------   Setup   ----------------------    
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

    # Load student tokenizer and feature extractor:
    tokenizer = WhisperTokenizerFast.from_pretrained(
        config.pretrained_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # NOTE: Because `language` and `task` have been set, the tokenizer will append the associated
    #       special tokens to the decoded sentence.
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.pretrained_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # Load student processor (to wrap the whole pipeline for saving):
    processor = WhisperProcessor.from_pretrained(
        config.pretrained_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )

    
    # Create the data collator that will be used to prepare the data for training:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=tokenizer,
                                                         feature_extractor=feature_extractor,
                                                         replace_padded_with_loss_mask_for_labels=True,
                                                         discard_first_bos_token=True)
    
    
    # Load the dataset and preprocess it:
    if config.smart_load:
        dataset_dict = smart_load_dataset_dict(config=config, processor=processor)
    else:
        print(f"Loading raw dataset `{config.dataset_name}` from Huggingface...")
        dataset_dict = load_dataset_dict(dataset_name=config.dataset_name)
        
        print(f"Preprocessing dataset `{config.dataset_name}`...")
        dataset_dict = preprocess_dataset(dataset_dict,  # type: ignore
                                          tokenizer=processor.tokenizer,  # type: ignore
                                          feature_extractor=processor.feature_extractor,  # type: ignore
                                          lowercase=config.lowercase,
                                          augment=config.data_augmentation)
    
    print("\n-----------------------\n")
    
    
    # Initialize the model from a pretrained checkpoint:
    print(f"Loading pretrained model `{config.pretrained_model_name_or_path}`...")
    if not config.experimental_train_implicit_lm:
        model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path)
    else:
        print("Enforcing the model to have zeroed cross-attention vectors for the encoder self-attention...")
        model = WhisperForConditionalGenerationZeroCrossAttention.from_pretrained(config.pretrained_model_name_or_path)
    
    # Freeze the encoder and/or decoder if specified in the config:
    assert not (config.freeze_encoder and config.freeze_decoder), \
        "Freezing both the encoder and the decoder would result in a model with " + \
        "no trainable parameters. Please set either `freeze_encoder` or `freeze_decoder` to `False`."
        
    if config.freeze_encoder:
        print("Freezing encoder...")
        model.freeze_encoder()  # type: ignore
    if config.freeze_decoder:
        print("Freezing decoder...")
        decoder = model.get_decoder()  # type: ignore
        for param in decoder.parameters():
            param.requires_grad = False
        decoder._requires_grad = False  # type: ignore
    
    
    # Set config parameters for training:
    if config.gradient_checkpointing:
        model.config.use_cache = False  # type: ignore
    
    
    # Set language and task for generation if not zero-shot. Also re-enable caching to speed-up evaluation:
    if config.zero_shot_eval:
        model.generate = partial(model.generate, language=None, task=None, use_cache=True)
        model.config.suppress_tokens = []
    else:
        model.generate = partial(model.generate, language=config.lang_name, task=config.task, use_cache=True)
    
    
    # Prepare training:
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    
    training_arguments_dict = dict(
        output_dir=config.model_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=True,
        fp16_full_eval=True,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        generation_num_beams=config.generation_num_beams,
        logging_strategy="steps",
        logging_first_step=True,
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
    
    if isinstance(config, TACFinetuneConfig):  # equivalent to `if tac`
        training_args = TACFinetuningTrainingArguments(languages_to_preserve=config.languages_to_preserve,
                                                       gamma_tac=config.gamma_tac,
                                                       **training_arguments_dict)
    else:
        training_args = Seq2SeqTrainingArguments(**training_arguments_dict)  # type: ignore
    
    # Define the compute_metrics function:
    compute_metrics = partial(compute_string_edit_metrics_fct,
                              processor=processor,
                              normalize=True)
    
    
    # Define callbacks:
    callbacks: List[TrainerCallback] = []
    
    if config.eval_first_step:
        callbacks.append(EvalFirstStepCallback())
    
    if config.early_stopping_patience != -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))  # type: ignore
    
    if config.log_preds_to_wandb:
        callbacks.append(WandbFinetuneCallback(config=config,
                                               processor=processor,
                                               eval_dataset=dataset_dict["validation"],  # type: ignore
                                               n_samples=config.n_samples_per_wandb_logging_step,
                                               log_raw_str=config.log_raw_str))
    
    
    # Create the trainer:
    trainer_args = dict(
        args=training_args,
        model=model,  # type: ignore
        train_dataset=dataset_dict["train"],  # type: ignore
        eval_dataset=dataset_dict["validation"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
        tokenizer=processor,  # use processor for saving  # type: ignore
        callbacks=callbacks
    )
    
    if tac:
        trainer = TACFinetuningTrainer(processor=processor, **trainer_args)
    else:
        trainer = Seq2SeqTrainer(**trainer_args)
    
    
    print("\n-----------------------\n")
    print("Starting training...")
    
    # Train the model:
    trainer.train()
    
    print("Training finished.")
    
    
    if config.save_final_model:
        # Save the model:
        final_model_dir = str(Path(config.model_dir) / "final")
        trainer.save_model(final_model_dir)
        print(f"Model saved to `{final_model_dir}`.")
    
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
