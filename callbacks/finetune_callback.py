from typing import Dict, Optional

import pandas as pd

import torch

from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import WhisperProcessor
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl
from datasets import Dataset

from callbacks.base_training_callback import BaseWandbTrainingCallback
from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from evaluation.string_edit_metrics import get_string_edit_metrics
from utils.finetune_config import FinetuneConfig
from utils.constants import GEN_MAX_LENGTH, LOSS_MASK_IDX, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WandbFinetuneCallback(BaseWandbTrainingCallback):
    """
    A callback for fine-tuning that logs the first `n_samples` of the evaluation dataset to W&B.
    """
    
    def __init__(self,
                 config: FinetuneConfig,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 log_raw_str: bool=False):
        super().__init__(config,
                         processor,
                         eval_dataset,
                         n_samples,
                         log_raw_str)
        
        assert isinstance(self.config, FinetuneConfig), "config must be `FinetuneConfig`"
        
        self.table_name = "sample_predictions-finetune"
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=processor.tokenizer,
                                                                  feature_extractor=processor.feature_extractor)
    
    
    def log_records_to_wandb(self) -> None:
        assert self.table_name is not None, "`table_name` must be set in child class"
        
        # Create a dataframe from the records:
        df = pd.DataFrame(self.records)
        df["wer"] = df["wer"].round(decimals=3)
        
        # Create a new wandb table:
        table_preds = self._wandb.Table(dataframe=df)
        self._wandb.log({self.table_name: table_preds})
        return
    
    
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               model: PreTrainedModel,
               logs: Optional[Dict[str, float]]=None,
               **kwargs):
        
        # Call `BaseWandbTrainingCallback`'s parent (`WandbCallback`) method `on_log` for basic logging:
        super(BaseWandbTrainingCallback, self).on_log(args, state, control, model, logs, **kwargs)  # type: ignore
        
        # Iterate through the first n samples:
        for idx, data in enumerate(self.eval_dataset):
            if idx >= self.n_samples:
                break
            
            # Log the original audio (should be done before call to DataCollator):
            self.log_audio_to_records(data)
            
            # Collate the data into batches of size 1:
            data = self.data_collator([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device)
            label_ids = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Generate the predictions:
            pred_ids = model.generate(input_features,
                                      max_length=GEN_MAX_LENGTH,
                                      num_beams=self.config.generation_num_beams)  # type: ignore
            
            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==LOSS_MASK_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Decode both the predictions and the labels:
            self.log_seq_to_records(label_ids, key="label", is_raw=False)
            self.log_seq_to_records(pred_ids, key="pred", is_raw=False)
            
            # Decode the predictions and labels without removing special tokens:
            if self.log_raw_str:
                self.log_seq_to_records(label_ids, key="label_raw", is_raw=True)
                self.log_seq_to_records(pred_ids, key="pred_raw", is_raw=True)
            
            # Retrieve the current label and prediction strings:
            curr_label_str = self.records["label"][-1]
            curr_pred_str = self.records["pred"][-1]
            
            # Compute the WER:
            self.records["wer"].append(100 * get_string_edit_metrics(references=[curr_label_str], predictions=[curr_pred_str])["wer"])
            
            # Add boolean flag to indicate whether the prediction is correct:
            self.records["is_correct"].append(curr_label_str == curr_pred_str)
            
            # Add information about the current training state:
            self.records["epoch"].append(state.epoch)
            self.records["step"].append(state.global_step)
        
        
        # Log the records to wandb:
        self.log_records_to_wandb()
        
        return
