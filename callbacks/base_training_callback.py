from abc import ABC, abstractmethod
from typing import Dict, Optional

from collections import defaultdict

import pandas as pd

import torch
from torch import Tensor

from transformers import (PreTrainedModel,
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
from utils.constants import DEFAULT_LABEL_STR_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseWandbTrainingCallback(WandbCallback, ABC):
    """
    Custom Huggingface's Trainer callback to log the progress results to W&B.
    """
    
    def __init__(self,
                 config: FinetuneConfig | DistilConfig,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 log_raw_str: bool=False):
        
        super(WandbCallback, self).__init__()
        
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
        
        # Save records as a class attribute. Doing so will allow to keep appending rows at each step,
        # and finally to relog the wandb tabel at the end of each step:
        self.records = defaultdict(list)
        
        self.table_name: Optional[str] = None
    
    
    @abstractmethod
    def get_predictions(self, model: PreTrainedModel, inputs) -> GenerateOutput | torch.Tensor:
        pass
    
    
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
        assert self.table_name is not None, "`table_name` must be set in child class"
        
        # Create a dataframe from the records:
        df = pd.DataFrame(self.records)
        df["wer"] = df["wer"].round(decimals=3)
        
        # Create a new wandb table:
        table_preds = self._wandb.Table(dataframe=df)
        self._wandb.log({self.table_name: table_preds})
        return
    
    
    @abstractmethod
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               model: PreTrainedModel,
               logs: Optional[Dict[str, float]]=None,
               **kwargs):
        pass