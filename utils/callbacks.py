from typing import Dict, Optional

from collections import defaultdict

import pandas as pd

import torch
from torch import Tensor

from transformers import (GenerationMixin,
                          PreTrainedModel,
                          WhisperProcessor,
                          TrainingArguments,
                          TrainerState,
                          TrainerControl)
from transformers.generation.utils import GenerateOutput
from transformers.integrations import WandbCallback
from datasets import Dataset
import evaluate

import wandb

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.finetune_config import FinetuneConfig
from utils.distil_config import DistilConfig
from utils.constants import DEFAULT_LABEL_STR_COL, GEN_MAX_LENGTH, PADDING_IDX, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WandbCustomCallback(WandbCallback):
    """
    Custom Huggingface's Trainer callback to log the progress results to W&B.
    """
    
    def __init__(self,
                 config: FinetuneConfig | DistilConfig,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 log_raw_str: bool=False):
        super().__init__()
        
        self.config = config
        assert isinstance(self.config, (FinetuneConfig, DistilConfig)), "config must be either `FinetuneConfig` or `DistilConfig`"
        self.is_distillation = isinstance(config, DistilConfig)
        
        self.processor = processor
        self.log_raw_str = log_raw_str
        
        # Convert to iterable dataset to be able to take samples:
        self.eval_dataset = eval_dataset
        self.eval_dataset = self.eval_dataset.filter(lambda x: x.strip(), input_columns=[DEFAULT_LABEL_STR_COL])
        self.eval_dataset.set_format("torch")
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        self.n_samples = n_samples
        self.wer_metric = evaluate.load("wer")
        
        # Save records as a class attribute. Doing so will allow to keep appending rows at each step,
        # and finally to relog the wandb tabel at the end of each step:
        self.records = defaultdict(list)
    
    
    def get_predictions(self, model: PreTrainedModel, inputs) -> GenerateOutput | torch.Tensor:
        if not self.is_distillation:
            pred_ids = model.generate(inputs,
                                      max_length=GEN_MAX_LENGTH,
                                      num_beams=self.config.generation_num_beams)  # type: ignore
        else:
            output = model.forward(**inputs)
            pred_ids = torch.argmax(output.logits, dim=-1)  # type: ignore
        
        return pred_ids
    
    
    def log_audio_to_records(self, data) -> None:
        audio = wandb.Audio(data["audio"]["array"], sample_rate=data["audio"]["sampling_rate"].item())  # type: ignore
        self.records["audio"].append(audio)
        return
    
    
    def log_seq_to_records(self,
                           tokenized_seq: GenerateOutput | Tensor,
                           key: str,
                           is_raw: bool=False) -> None:
        curr_label_str = self.processor.tokenizer.batch_decode(tokenized_seq, skip_special_tokens=not(is_raw), normalize=not(is_raw))[0]  # type: ignore
        self.records[key].append("".join(curr_label_str))
        return
    
    
    def log_records_to_wandb(self) -> None:
        # Create a dataframe from the records:
        df = pd.DataFrame(self.records)
        df["wer"] = df["wer"].round(decimals=3)
        
        # Create a new wandb table:
        table_preds = self._wandb.Table(dataframe=df)
        self._wandb.log({"sample_predictions": table_preds})
        
        return
    
    
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               model: PreTrainedModel,
               logs: Optional[Dict[str, float]]=None,
               **kwargs):
        
        super().on_log(args, state, control, model, logs, **kwargs)
                
        # Iterate through the first n samples:
        for idx, data in enumerate(self.eval_dataset):
            if idx >= self.n_samples:
                break
            
            # Log the original audio (should be done before call to DataCollator):
            self.log_audio_to_records(data)
            
            # Collate the data into batches of size 1:
            data = self.data_collator([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            inputs = data["input_features"].to(device)
            label_ids = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Generate the predictions:
            pred_ids = self.get_predictions(model, inputs)
            
            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==PADDING_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Decode both the predictions and the labels:
            self.log_seq_to_records(label_ids, key="label_str", is_raw=False)
            self.log_seq_to_records(pred_ids, key="pred_str", is_raw=False)        
            
            # Decode the predictions and labels without removing special tokens:
            if self.log_raw_str:
                self.log_seq_to_records(label_ids, key="label_str_raw", is_raw=True)
                self.log_seq_to_records(pred_ids, key="pred_str_raw", is_raw=True)
            
            # Retrieve the current label and prediction strings:
            curr_label_str = self.records["label_str"][-1]
            curr_pred_str = self.records["pred_str"][-1]
            
            # Compute the WER:
            self.records["wer"].append(100 * self.wer_metric.compute(references=[curr_label_str], predictions=[curr_pred_str]))  # type: ignore
            
            # Add boolean flag to indicate whether the prediction is correct:
            self.records["is_correct"].append(curr_label_str == curr_pred_str)
            
            # Add information about the current training state:
            self.records["epoch"].append(state.epoch)
            self.records["step"].append(state.global_step)
        
        
        # Log the records to wandb:
        self.log_records_to_wandb()
        
        return
