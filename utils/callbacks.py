from typing import Dict, Optional

from collections import defaultdict

import pandas as pd

import torch

from transformers import (GenerationMixin,
                          GenerationMixin,
                          WhisperProcessor,
                          TrainingArguments,
                          TrainerState,
                          TrainerControl)
from transformers.integrations import WandbCallback
from datasets import Dataset
import evaluate

import wandb

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.config import Config
from utils.constants import DEFAULT_LABEL_STR_COL, GEN_MAX_LENGTH, PADDING_IDX, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WandbCustomCallback(WandbCallback):
    """
    Custom Huggingface's Trainer callback to log the progress results to W&B.
    """
    
    def __init__(self,
                 config: Config,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 log_raw_str: bool=False):
        super().__init__()
        self.config = config
        self.processor = processor
        self.log_raw_str = log_raw_str
        
        # Convert to iterable dataset to be able to take samples:
        self.eval_dataset = eval_dataset
        self.eval_dataset = self.eval_dataset.filter(lambda x: x.strip(), input_columns=[DEFAULT_LABEL_STR_COL])
        self.eval_dataset.set_format("torch")
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        self.n_samples = n_samples
        self.wer_metric = evaluate.load("wer")
    
    
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               model: GenerationMixin,
               logs: Optional[Dict[str, float]]=None,
               **kwargs):
        
        super().on_log(args, state, control, model, logs, **kwargs)
        
        records = defaultdict(list)
        
        # Iterate through the first n samples:
        for idx, data in enumerate(self.eval_dataset):
            if idx >= self.n_samples:
                break
            
            # Log the original audio (should be done before call to DataCollator):
            audio = wandb.Audio(data["audio"]["array"], sample_rate=data["audio"]["sampling_rate"].item())  # type: ignore
            records["audio"].append(audio)
            
            # Collate the data into batches of size 1:
            data = self.data_collator([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            inputs = data["input_features"].to(device)
            label_ids = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            
            # Generate the predictions:
            pred_ids = model.generate(inputs,
                                      max_length=GEN_MAX_LENGTH,
                                      num_beams=self.config.generation_num_beams)

            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==PADDING_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Decode the predictions:
            curr_pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)[0]  # type: ignore
            curr_pred_str = "".join(curr_pred_str)
            records["pred_str"].append(curr_pred_str)
            
            # Decode the labels:
            curr_label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=True)[0]  # type: ignore
            curr_label_str = "".join(curr_label_str)
            records["label_str"].append(curr_label_str)
            
            # Decode the predictions and labels without removing special tokens:
            if self.log_raw_str:
                curr_pred_str_raw = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False, normalize=True)[0]  # type: ignore
                curr_pred_str_raw = "".join(curr_pred_str_raw)
                records["pred_str_raw"].append(curr_pred_str_raw)
                
                curr_label_str_raw = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=False, normalize=False)[0]  # type: ignore
                curr_label_str_raw = "".join(curr_label_str_raw)
                records["label_str_raw"].append(curr_label_str_raw)
            
            # Compute the WER:
            records["wer"].append(100 * self.wer_metric.compute(references=[curr_label_str], predictions=[curr_pred_str]))  # type: ignore
            
            # Add boolean flag to indicate whether the prediction is correct:
            records["is_correct"].append(curr_label_str == curr_pred_str)
            
            # Add information about the current training state:
            records["epoch"].append(state.epoch)
            records["step"].append(state.global_step)
        
        
        # Create a dataframe from the records:
        df = pd.DataFrame(records)
        df["wer"] = df["wer"].round(2)
        
        
        # Create a new wandb table:
        table_preds = self._wandb.Table(dataframe=df)
        self._wandb.log({"sample_predictions": table_preds})
        
        return
