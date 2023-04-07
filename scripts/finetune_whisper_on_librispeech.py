import typer

from dataclasses import asdict
import wandb

from transformers import (WhisperForConditionalGeneration,
                          WhisperProcessor,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          WhisperTokenizer,
                          WhisperFeatureExtractor)

from dataloader.dataloader import convert_dataset_dict_to_torch, load_dataset_dict, shuffle_dataset_dict
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.preprocessing import preprocess_dataset
from dataloader.utils import sample_from_dataset
from models.callbacks import ShuffleCallback
from evaluation.metrics import compute_wer
from utils.config import parse_yaml_config


N_SAMPLES_EVAL = 10


def main(config_filepath: str):
    config = parse_yaml_config(config_filepath)

    # ----------------------- W&B -----------------------
    # login to weights & biases
    wandb.login()

    # initialize tracking the experiment
    wandb.init(project="whisper_finetuning",
               job_type="fine-tuning",
               config=asdict(config))


    # Load feature extractor, tokenizer and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.pretrained_model_name_or_path
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        language=config.lang_name,
        task="transcribe",
        model_max_length=225
    )

    processor = WhisperProcessor.from_pretrained(
        config.pretrained_model_name_or_path,
        task="transcribe",
        model_max_length=225
        )


    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    dataset_dict = load_dataset_dict(dataset_name="librispeech")
    
    dataset_dict = preprocess_dataset(dataset_dict, feature_extractor=feature_extractor)
    dataset_dict = shuffle_dataset_dict(dataset_dict)
    dataset_dict = convert_dataset_dict_to_torch(dataset_dict)
    
    # load a sample dataset that will be used to log the training progress
    samples_dataset = sample_from_dataset(dataset_dict["test"], n_samples=N_SAMPLES_EVAL)

    # initialize the model from a pretrained checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path, use_cache=False)
    model.config.forced_decoder_ids = None  # type: ignore
    model.config.suppress_tokens = []  # type: ignore
    model.config.use_cache = False  # type: ignore

    # create the training arguments required to train the model
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.model_dir,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        save_total_limit=2,
        warmup_steps=25,
        max_steps=100,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_bnb_8bit",
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25,
        eval_steps=25,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False, 
        ignore_data_skip=True
    )
    
    callbacks = [ShuffleCallback()]

    # create the trainer class that will be used to train the model
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,  # type: ignore
        train_dataset=dataset_dict["train"],
        eval_dataset=samples_dataset,  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_wer,  # type: ignore
        tokenizer=tokenizer,
        callbacks=callbacks  # type: ignore
    )

    #save the model and processor before we begin training
    model.save_pretrained(training_args.output_dir)  # type: ignore
    processor.save_pretrained(training_args.output_dir)


    # train the model
    trainer.train()

    # finish tracking the experiment
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
