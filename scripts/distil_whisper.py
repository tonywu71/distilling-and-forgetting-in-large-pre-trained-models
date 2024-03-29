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

from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from optimum.bettertransformer import BetterTransformer

import wandb

from callbacks.distillation_callback import WandbDistillationCallback
from callbacks.eval_first_step_callback import EvalFirstStepCallback
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.collator_distil import DataCollatorWithPaddingForSeqLevelDistillation
from dataloader.dataset_loader import load_dataset_dict
from dataloader.preprocessing_train.preprocessing import preprocess_dataset
from dataloader.smart_load_dataset_dict import smart_load_dataset_dict
from evaluation.wer_metric import compute_string_edit_metrics_fct
from k_beam_search.smart_load_k_beam_search import smart_load_dataset_with_k_beam_search
from normalization.whisper_normalization import get_whisper_normalizer
from trainer.teacher_filtering import filter_teacher_outputs
from trainer.teacher_postprocessing import postprocess_teacher_outputs
from trainer.distillation_word_level import DistillationWordLevelTrainingArguments, DistillationWordLevelTrainer
from trainer.distillation_seq_level import DistillationSeqLevelTrainingArguments, DistillationSeqLevelTrainer
from utils.distil_config import DistilConfig
from utils.file_io import fix_model_dir_conflicts
from utils.sanity_checks import assert_if_distillation_tokenizers_match

from utils.constants import GEN_MAX_LENGTH



def main(config_filepath: str = typer.Argument(..., help="Path to the YAML config file."),
         teacher_caching_batch_size: int = typer.Option(16, help="Batch size for caching teacher outputs."),
         end_after_caching: bool = typer.Option(False, help="Whether to end the script after caching. " + \
                                                "Used when the maximum compute time is too short to perform distillation right after caching"),
         debug: bool = typer.Option(False, help="Whether to run in debug mode or not.")):
    """
    Distil Whisper based on the provided config file.
    """

    # Get the device:
    if torch.cuda.is_available():
        device = "cuda:0"
        teacher_dtype = torch.float16  # see https://huggingface.co/learn/audio-course/chapter5/evaluation?fw=pt
    elif torch.backends.mps.is_available():  # for Apple Silicon
        device = torch.device('mps')
        teacher_dtype = torch.float32  # float16 not supported by MPS
    else:
        device = "cpu"
        teacher_dtype = torch.float32
    
    # --------------------   Load config   --------------------
    config = DistilConfig.from_yaml(config_filepath)
    
    # Sanity check:
    assert_if_distillation_tokenizers_match(config)
    
    is_seq_level = config.method_distil in ["seq_level_uniform", "seq_level_ranked"]
    
    if is_seq_level and config.distillation_num_beams > 1:
        print("\n-----------------------\n")
        print(f"Sequence-level distillation will be performed. Although the batch size is set to {config.batch_size}, " + \
              f"because {config.distillation_num_beams} beams will be used for distillation, " + \
              f"the actual batch size will {config.batch_size * config.distillation_num_beams}.")
        print("\n-----------------------\n")

    
    # If a previous run has its checkpoints saved in the same directory,
    # add a timestamp to the model directory. This is to avoid overwriting
    # previous models. Note that `config` is modified in-place.
    fix_model_dir_conflicts(config)
    
    # Prepare tags for W&B:
    list_tags = [config.dataset_name,
                 config.method_distil]
    
    if is_seq_level:
        if config.distillation_num_beams == 1:
            list_tags.append("1_best_kd")
        if config.distillation_num_beams > 1:
            list_tags.append("k_best_kd")
    
    if end_after_caching:
        print("\n-----------------------\n")
        print("Ending script after caching is enabled. Distillation will not be performed.")
        print("\n-----------------------\n")
        list_tags.append("caching")
    
    # -----------------------   W&B   -----------------------
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_TRAINING"],
               job_type="distillation",
               tags=list_tags,
               name=config.experiment_name,
               config=asdict(config),
               mode="disabled" if debug else None)
    
    
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

    # Load student processor (to wrap the whole pipeline for saving):
    student_processor = WhisperProcessor.from_pretrained(
        config.student_model_name_or_path,
        language=config.lang_name,
        task=config.task
    )

    # NOTE: Because `language` and `task` have been set, the tokenizer will append the associated
    #       special tokens to the decoded sentence.
    
    # Create the data collator that will be used to prepare the data for training:
    data_collator_args = dict(
        tokenizer=student_processor.tokenizer,
        feature_extractor=student_processor.feature_extractor,
        return_attention_mask=True,
        replace_padded_with_loss_mask_for_labels=True,
        discard_first_bos_token=True
    )
    if not is_seq_level:  # If word-level...
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(**data_collator_args)
    else:
        data_collator = DataCollatorWithPaddingForSeqLevelDistillation(**data_collator_args,
                                                                       distillation_k_beam=config.distillation_num_beams)
    
    # Load the dataset and preprocess it:
    if config.smart_load:
        dataset_dict = smart_load_dataset_dict(config=config, processor=student_processor)
    else:
        print(f"Loading raw dataset `{config.dataset_name}` from Huggingface...")
        dataset_dict = load_dataset_dict(dataset_name=config.dataset_name)
        
        print(f"Preprocessing dataset `{config.dataset_name}`...")
        dataset_dict = preprocess_dataset(dataset_dict,
                                          tokenizer=student_processor.tokenizer,
                                          feature_extractor=student_processor.feature_extractor,
                                          augment=config.data_augmentation)
    
    print("\n-----------------------\n")
    
    if is_seq_level:
        # Overwrite `dataset_dict` with the pre-computed K-beam search outputs from the teacher model:
        dataset_dict = smart_load_dataset_with_k_beam_search(config=config,
                                                             dataset_dict=dataset_dict,
                                                             teacher_caching_batch_size=teacher_caching_batch_size)
        print("\n-----------------------\n")
    

    if config.unsupervised_word_level:
        # TODO: Implementation is a bit hacky. Refactor this.
        print("Word-level distillation will be performed in an unsupervised manner.")
        
        # Hotfix to use `smart_load_dataset_with_k_beam_search`:
        config.method_distil = "seq_level_uniform"
        config.distillation_num_beams = 1

        # Overwrite `dataset_dict` with the pre-computed K-beam search outputs from the teacher model:
        dataset_dict = smart_load_dataset_with_k_beam_search(config=config,
                                                             dataset_dict=dataset_dict,
                                                             teacher_caching_batch_size=teacher_caching_batch_size)
        
        # NOTE: We won't replace the `labels` column with the `teacher_sequences` column yet because we want to
        #       post-process and/or filter the teacher outputs first.
        print("\n-----------------------\n")


    if end_after_caching:
        print("Ending script after caching teacher outputs.")
        wandb.finish()
        return
    
    
    # NOTE: Because the validation splits are subsampled, we should run the full evaluation on the validation split
    #       manually after traning.
    if config.dataset_name == "librispeech_clean_100h":
        print("Subsampling the 100h LibriSpeech validation split to 50% of its original size for faster evaluation...")
        dataset_dict["validation"] = dataset_dict["validation"].select(range(dataset_dict["validation"].num_rows // 2))
    elif config.dataset_name == "ami_100h":
        print("Subsampling the 100h AMI validation split to 20% of its original size for faster evaluation...")
        dataset_dict["validation"] = dataset_dict["validation"].select(range(dataset_dict["validation"].num_rows // 5))
    

    # Filtering/post-processing:
    if is_seq_level and (config.postprocess_teacher or config.strip_teacher):
        dataset_dict = postprocess_teacher_outputs(dataset_dict=dataset_dict, config=config)
    
    if config.max_exceeding_tokens or config.max_teacher_gzip_ratio or config.max_ratio_instant_tokens:
        dataset_dict = filter_teacher_outputs(dataset_dict=dataset_dict, config=config)
    

    if config.unsupervised_word_level:
        # TODO: Implementation is a bit hacky. Refactor this.
        
        # We will replace the `labels` column with the `teacher_sequences` column (train split only):
        dataset_dict["train"] = dataset_dict["train"].remove_columns(["labels"])
        dataset_dict["train"] = dataset_dict["train"].rename_column("teacher_sequences", "labels")

        # Restore the original config:
        config.method_distil = "word_level"
        config.distillation_num_beams = None
    

    # Initialize the models from pretrained checkpoints:
    if config.method_distil == "word_level":
        print(f"Loading teacher model `{config.teacher_model_name_or_path}`...")
        teacher_model = WhisperForConditionalGeneration.from_pretrained(config.teacher_model_name_or_path).to(device).to(teacher_dtype)
        if torch.cuda.is_available():
            print("CUDA is available. Transforming the teacher model to use the BetterTransformer...")
            teacher_model = BetterTransformer.transform(teacher_model)
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
        student_model.freeze_encoder()
    if config.freeze_decoder:
        print("Freezing the student's decoder...")
        decoder = student_model.get_decoder()
        for param in decoder.parameters():
            param.requires_grad = False
        decoder._requires_grad = False
    
    
    # Set config parameters for training:
    if config.gradient_checkpointing:
        student_model.config.use_cache = False 
    
    
    # Set language and task for generation if not zero-shot. Also re-enable caching to speed-up evaluation:
    if config.zero_shot_eval:
        student_model.generate = partial(student_model.generate, language=None, task=None, use_cache=True)
    else:
        student_model.generate = partial(student_model.generate, language=config.lang_name, task=config.task, use_cache=True)
    
    
    # Same for the teacher model.
    if teacher_model is not None:  # ignore teacher model if not used
        teacher_model.generate = partial(teacher_model.generate, language=config.lang_name, task=config.task, use_cache=True)
        # NOTE: The teacher model generation is NOT zero-shot.
    
    
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
        load_best_model_at_end=False if config.save_total_limit == 1 else True,
        metric_for_best_model="wer",
        greater_is_better=False,  # the lower the WER, the better
        report_to="wandb"
    )
    
    if config.method_distil == "word_level":
        training_args = DistillationWordLevelTrainingArguments(**training_arguments_dict)
    elif config.method_distil in ["seq_level_uniform", "seq_level_ranked"]:
        training_args = DistillationSeqLevelTrainingArguments(**training_arguments_dict)
    else:
        raise ValueError(f"Invalid distillation method `{config.method_distil}`.")
        
    
    # Define the compute_metrics function:
    compute_metrics = partial(compute_string_edit_metrics_fct,
                              processor=student_processor,
                              whisper_norm=get_whisper_normalizer(language=config.lang_name))
    
    
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
        train_dataset=dataset_dict["train"],  # type: ignore
        eval_dataset=dataset_dict["validation"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
        tokenizer=student_processor,  # use processor for saving  # type: ignore
        callbacks=callbacks
    )
    
    if isinstance(training_args, DistillationWordLevelTrainingArguments):
        distillation_trainer = DistillationWordLevelTrainer(teacher_model=teacher_model, **trainer_args)
    elif isinstance(training_args, DistillationSeqLevelTrainingArguments):
        distillation_trainer = DistillationSeqLevelTrainer(**trainer_args)
    else:
        raise ValueError(f"Invalid training arguments type `{type(training_args)}`.")
    
    
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
