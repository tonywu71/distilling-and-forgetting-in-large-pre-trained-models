import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

import torch
assert torch.cuda.is_available(), "This script requires a GPU."
device = torch.device("cuda:0")

from utils.initialize import initialize_env, print_envs
initialize_env()

from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pprint

from transformers import (WhisperForConditionalGeneration,
                          WhisperProcessor,
                          EarlyStoppingCallback,
                          TrainerCallback)

import wandb

from callbacks.distillation_callback import WandbDistillationCallback
from callbacks.eval_first_step_callback import EvalFirstStepCallback
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataloader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import preprocess_dataset
from dataloader.smart_load_dataset_dict import smart_load_dataset_dict
from evaluation.wer_metric import compute_wer_fct_distil
from k_beam_search.smart_load_k_beam_search import smart_load_dataset_with_k_beam_search
from trainer.distillation import DistillationTrainer, DistillationTrainingArguments
from utils.distil_config import DistilConfig
from utils.file_io import fix_model_dir_conflicts
from utils.sanity_checks import distillation_sanity_check



def main(config_filepath: str):
    """
    Distil the Whisper model on the LibriSpeech dataset.
    """
    # --------------------   Load config   --------------------
    config = DistilConfig.from_yaml(config_filepath)
    distillation_sanity_check(config)
    
    is_seq_level = config.method in ["seq_level_k_best_uniform", "seq_level_k_best_ranked"]
    
    if is_seq_level:
        print(f"Sequence-level distillation will be performed. Although the batch size is set to {config.batch_size}, " + \
              f", because {config.distillation_num_beams} beams will be used for distillation, " + \
              f"the actual batch size will {config.batch_size * config.distillation_num_beams}.")
    
    # If a previous run has its checkpoints saved in the same directory,
    # add a timestamp to the model directory. This is to avoid overwriting
    # previous models. Note that `config` is modified in-place.
    fix_model_dir_conflicts(config)
    
    # Prepare tags for W&B:
    list_tags = [config.method]
    if config.is_hpt:
        list_tags.append("hpt")
    
    # -----------------------   W&B   -----------------------
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT"],
               job_type="distillation",
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
    
    # Load student processor (contains both tokenizer and feature extractor):
    student_processor = WhisperProcessor.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # Create the data collator that will be used to prepare the data for training:
    if not is_seq_level:  # If word-level...
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=student_processor,
                                                             return_attention_mask_labels=True,
                                                             add_k_beam_features=False)
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=student_processor,
                                                             return_attention_mask_labels=True,
                                                             add_k_beam_features=True)
    
    # Load the dataset and preprocess it:
    if config.smart_load:
        dataset_dict = smart_load_dataset_dict(config=config, processor=student_processor)
    else:
        print(f"Loading raw dataset `{config.dataset_name}` from Huggingface...")
        dataset_dict = load_dataset_dict(dataset_name=config.dataset_name)
        
        print(f"Preprocessing dataset `{config.dataset_name}`...")
        dataset_dict = preprocess_dataset(dataset_dict,  # type: ignore
                                          tokenizer=student_processor.tokenizer,  # type: ignore
                                          feature_extractor=student_processor.feature_extractor,  # type: ignore
                                          augment=config.data_augmentation)
    
    print("\n-----------------------\n")
    
    if is_seq_level:
        # Overwrite `dataset_dict` with the pre-computed K-beam search outputs from the teacher model:
        dataset_dict = smart_load_dataset_with_k_beam_search(config=config,
                                                             dataset_dict=dataset_dict)
    
        # Note: Technically, the K-beam search features are not needed for the word-level distillation. However,
        #       we still load them for simplicity and because they are needed for `WandbDistillationCallback`.
    
        print("\n-----------------------\n")
    
    # Initialize the models from pretrained checkpoints:
    if not is_seq_level:  # If word-level...
        print(f"Loading teacher model `{config.teacher_model_name_or_path}`...")
        teacher_model = WhisperForConditionalGeneration.from_pretrained(config.teacher_model_name_or_path).to(device)  # type: ignore
        # Freeze the teacher model:
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model._requires_grad = False
    else:
        teacher_model = None  # the teacher model is not needed for sequence-level distillation as we already have its K-beam search outputs loaded
    
    print(f"Loading student model `{config.student_model_name_or_path}`...")
    student_model = WhisperForConditionalGeneration.from_pretrained(config.student_model_name_or_path).to(device)  # type: ignore
    
    
    # Freeze the student's encoder and/or decoder if specified in the config:
    assert not (config.freeze_encoder and config.freeze_decoder), \
        "Freezing both the encoder and the decoder would result in a model with " + \
        "no trainable parameters. Please set either `freeze_encoder` or `freeze_decoder` to `False`."
        
    if config.freeze_encoder:
        print("Freezing the student's encoder...")
        student_model.freeze_encoder()  # type: ignore
    if config.freeze_decoder:
        print("Freezing the student's decoder...")
        decoder = student_model.get_decoder()  # type: ignore
        for param in decoder.parameters():
            param.requires_grad = False
        decoder._requires_grad = False  # type: ignore
    
    
    # Notes:
    # - The Whisper model has token ids that are forced as model outputs before autoregressive generation is started (forced_decoder_ids).
    #   These token ids control the transcription language and task for zero-shot ASR. If `zero_shot` is enabled in config, we will set
    #   these ids to None, as we will train the model to predict the correct language and task (which are provided in the tokenized input).
    # - There are also tokens that are completely suppressed during generation (suppress_tokens). These tokens have their log probabilities
    #   set to -inf, such that they are never sampled. We'll override these tokens to an empty list, meaning no tokens are suppressed.
    for model in [teacher_model, student_model]:
        if model is not None:  # ignore teacher model if not used
            if config.zero_shot:
                model.config.forced_decoder_ids = None
            else:
                model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=config.lang_name, task=config.task)  # type: ignore
            model.config.suppress_tokens = []
    
    # Since only the student model is trained, we can keep caching for the teacher model:
    if config.gradient_checkpointing:
        student_model.config.use_cache = False
    
    
    # Prepare training:
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = DistillationTrainingArguments(
        method=config.method,
        alpha_ce=config.alpha_ce,
        temperature=config.temperature,
        distillation_num_beams=config.distillation_num_beams,
        beta_decay=config.beta_decay,
        output_dir=config.model_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=True,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=not(is_seq_level),  # keep the K-beam features if sequence-level, remove them if word-level
        load_best_model_at_end=True,
        metric_for_best_model="wer" if not is_seq_level else "eval_loss",
        greater_is_better=False,  # the lower the WER, the better (same for the loss)
        report_to="wandb"  # type: ignore
    )
    
    
    # Define the compute_metrics function:
    if not is_seq_level:  # If word-level...
        compute_wer = partial(compute_wer_fct_distil,
                              processor=student_processor,
                              normalize=True,
                              log_string_edit_metrics_on_wandb=True)
    else:  # If sequence-level...
        pass  # `compute_metrics` will be set to `None` in the trainer
    
    # Define callbacks:
    callbacks: List[TrainerCallback] = []
    
    if config.eval_first_step:
        callbacks.append(EvalFirstStepCallback())
    
    if config.early_stopping_patience != -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))  # type: ignore
    
    if config.log_preds_to_wandb:
        callbacks.append(WandbDistillationCallback(config=config,
                                                   processor=student_processor,
                                                   eval_dataset=dataset_dict["validation"],  # type: ignore
                                                   n_samples=config.n_samples_per_wandb_logging_step,
                                                   teacher_model=teacher_model,  # should be None if word-level distillation
                                                   log_raw_str=config.log_raw_str))
    
    
    # Create the trainer:
    distillation_trainer = DistillationTrainer(
        args=training_args,
        model=student_model,  # type: ignore
        student_tokenizer=student_processor.tokenizer,
        teacher_model=teacher_model,
        train_dataset=dataset_dict["train"],  # type: ignore
        eval_dataset=dataset_dict["validation"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_wer if not is_seq_level else None,
        tokenizer=student_processor,  # type: ignore
        callbacks=callbacks
    )
    
    print("Starting distillation...")
        
    # Distil the model:
    distillation_trainer.train()
    
    print("Distillation finished.")
    
    # Save the model:
    final_model_dir = Path(config.model_dir) / "final"
    distillation_trainer.save_model(final_model_dir)
    
    print(f"Model saved to `{final_model_dir}`.")
    
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
