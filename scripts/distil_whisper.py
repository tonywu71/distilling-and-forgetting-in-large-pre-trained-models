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

import torch
from transformers.models.whisper import (WhisperForConditionalGeneration,
                                         WhisperTokenizerFast,
                                         WhisperFeatureExtractor,
                                         WhisperProcessor)
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback

import wandb

from callbacks.distillation_callback import WandbDistillationCallback
from callbacks.eval_first_step_callback import EvalFirstStepCallback
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.dataset_loader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import preprocess_dataset
from dataloader.smart_load_dataset_dict import smart_load_dataset_dict
from k_beam_search.smart_load_k_beam_search import smart_load_dataset_with_k_beam_search
from evaluation.wer_metric import compute_string_edit_metrics_fct
from trainer.distillation import DistillationTrainer, DistillationTrainingArguments
from trainer.tac_distillation import TACDistillationTrainer, TACDistillationTrainingArguments
from utils.constants import GEN_MAX_LENGTH
from utils.distil_config import DistilConfig
from utils.file_io import fix_model_dir_conflicts
from utils.sanity_checks import assert_if_distillation_tokenizers_match
from utils.tac_distil_config import TACDistilConfig



def main(config_filepath: str = typer.Argument(..., help="Path to the YAML config file."),
         tac: bool = typer.Option(False, help="Whether to use Task Alignment Consolidation or not. " + \
                                              "Flag should be used if only if the config file is for TAC distillation.")):
    """
    Distil Whisper based on the provided config file.
    """
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # --------------------   Load config   --------------------
    if tac:
        config = TACDistilConfig.from_yaml(config_filepath)
    else:
        config = DistilConfig.from_yaml(config_filepath)
    
    # Sanity check:
    assert_if_distillation_tokenizers_match(config)
    
    is_seq_level = config.method_distil in ["seq_level_uniform", "seq_level_ranked"]
    
    if is_seq_level:
        print(f"Sequence-level distillation will be performed. Although the batch size is set to {config.batch_size}, " + \
              f"because {config.distillation_num_beams} beams will be used for distillation, " + \
              f"the actual batch size will {config.batch_size * config.distillation_num_beams}.")
    
    # If a previous run has its checkpoints saved in the same directory,
    # add a timestamp to the model directory. This is to avoid overwriting
    # previous models. Note that `config` is modified in-place.
    fix_model_dir_conflicts(config)
    
    # Prepare tags for W&B:
    list_tags = [config.dataset_name,
                 config.method_distil]
    if config.is_hpt:
        list_tags.append("hpt")
    if tac:
        list_tags.append("tac")
    
    # -----------------------   W&B   -----------------------
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_TRAINING"],
               job_type="distillation_with_tac" if tac else "distillation",
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
    student_tokenizer = WhisperTokenizerFast.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # NOTE: Because `language` and `task` have been set, the tokenizer will append the associated
    #       special tokens to the decoded sentence.
    
    student_feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    # Load student processor (to wrap the whole pipeline for saving):
    student_processor = WhisperProcessor.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )
    
    
    # Create the data collator that will be used to prepare the data for training:
    if not is_seq_level:  # If word-level...
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=student_tokenizer,
                                                             feature_extractor=student_feature_extractor,
                                                             return_attention_mask=True,
                                                             replace_padded_with_loss_mask_for_labels=True,
                                                             discard_first_bos_token=True)
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=student_tokenizer,
                                                             feature_extractor=student_feature_extractor,
                                                             return_attention_mask=True,
                                                             replace_padded_with_loss_mask_for_labels=True,
                                                             discard_first_bos_token=True,
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
        print("\n-----------------------\n")
    
    
    if config.dataset_name == "ami_100h":
        print("Subsampling the 100h AMI validation split to 10% of its original size for faster evaluation...")
        dataset_dict["validation"] = dataset_dict["validation"].select(range(dataset_dict["validation"].num_rows // 10))
    
    
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
    
    
    # Set config parameters for training:
    if config.gradient_checkpointing:
        student_model.config.use_cache = False  # type: ignore
    
    
    # Set language and task for generation if not zero-shot. Also re-enable caching to speed-up evaluation:
    if config.zero_shot_eval:
        student_model.generate = partial(student_model.generate, language=None, task=None, use_cache=True)
    else:
        student_model.generate = partial(student_model.generate, language=config.lang_name, task=config.task, use_cache=True)
    
    
    # Same for the teacher model.
    if teacher_model is not None:  # ignore teacher model if not used
        teacher_model.generate = partial(teacher_model.generate, language=config.lang_name, task=config.task, use_cache=True)
        # NOTE: The teacher model geneartion is NOT zero-shot.
    
    
    # Prepare training:
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    
    training_arguments_dict = dict(
        method_distil=config.method_distil,
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
        fp16_full_eval=True,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        generation_num_beams=config.generation_num_beams,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_LENGTH,
        remove_unused_columns=not(is_seq_level),  # keep the K-beam features if sequence-level, remove them if word-level
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,  # the lower the WER, the better
        report_to="wandb"  # type: ignore
    )
    
    if isinstance(config, TACDistilConfig):  # equivalent to `if tac`
        training_args = TACDistillationTrainingArguments(languages_to_preserve=config.languages_to_preserve,
                                                         method_tac=config.method_tac,
                                                         gamma_tac=config.gamma_tac,
                                                         **training_arguments_dict)
    else:
        training_args = DistillationTrainingArguments(**training_arguments_dict)  # type: ignore
    
    # Define the compute_metrics function:
    compute_metrics = partial(compute_string_edit_metrics_fct,
                              processor=student_processor,
                              normalize=True)
    
    
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
    trainer_args = dict(
        args=training_args,
        model=student_model,  # type: ignore
        student_processor=student_processor,
        teacher_model=teacher_model,
        train_dataset=dataset_dict["train"],  # type: ignore
        eval_dataset=dataset_dict["validation"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
        tokenizer=student_processor,  # use processor for saving  # type: ignore
        callbacks=callbacks
    )
    
    if tac:
        distillation_trainer = TACDistillationTrainer(**trainer_args)
    else:
        distillation_trainer = DistillationTrainer(**trainer_args)
    
    
    print("\n-----------------------\n")
    print("Starting distillation...")
        
    # Distil the model:
    distillation_trainer.train()
    
    print("Distillation finished.")
    
    if config.save_final_model:
        # Save the model:
        final_model_dir = str(Path(config.model_dir) / "final")
        distillation_trainer.save_model(final_model_dir)
        print(f"Model saved to `{final_model_dir}`.")
    
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
