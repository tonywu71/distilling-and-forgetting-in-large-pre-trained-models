from typing import Dict, Optional

import pandas as pd

import torch

from transformers import (PreTrainedModel,
                          WhisperProcessor,
                          TrainingArguments,
                          TrainerState,
                          TrainerControl)
from datasets import Dataset

from callbacks.base_training_callback import BaseWandbTrainingCallback
from utils.finetune_config import FinetuneConfig
from utils.distil_config import DistilConfig
from utils.constants import GEN_MAX_LENGTH, PADDING_IDX, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WandbDistillationCallback(BaseWandbTrainingCallback):
    def __init__(self,
                 config: FinetuneConfig | DistilConfig,
                 teacher_model: PreTrainedModel,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 log_raw_str: bool=False):
        super().__init__(config,
                         processor,
                         eval_dataset,
                         n_samples,
                         log_raw_str)
        self.teacher_model = teacher_model
        self.table_name = "sample_predictions-distill"
        assert isinstance(self.config, DistilConfig), "config must be `DistilConfig`"
    
    
    def log_records_to_wandb(self) -> None:
        assert self.table_name is not None, "`table_name` must be set in child class"
        
        # Create a dataframe from the records:
        df = pd.DataFrame(self.records)
        df["wer_student"] = df["wer_student"].round(decimals=3)
        
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
        # Note: `model` corresponds to the student model in this method.
        
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
            pred_ids_student = self.teacher_model.generate(input_features,
                                                           max_length=GEN_MAX_LENGTH,
                                                           num_beams=self.config.generation_num_beams)  # type: ignore
            pred_ids_teacher = self.teacher_model.generate(input_features,
                                                           max_length=GEN_MAX_LENGTH,
                                                           num_beams=self.config.generation_num_beams)  # type: ignore
            
            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==PADDING_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Decode both the predictions and the labels:
            self.log_seq_to_records(label_ids, key="label", is_raw=False)
            self.log_seq_to_records(pred_ids_teacher, key="pred_teacher", is_raw=False)
            self.log_seq_to_records(pred_ids_student, key="pred_student", is_raw=False)
            
            
            # Decode the predictions and labels without removing special tokens:
            if self.log_raw_str:
                self.log_seq_to_records(label_ids, key="label_raw", is_raw=True)
                self.log_seq_to_records(pred_ids_teacher, key="pred_teacher_raw", is_raw=True)
                self.log_seq_to_records(pred_ids_student, key="pred_student_raw", is_raw=True)
            
            # Retrieve the current label and prediction strings:
            curr_label_str = self.records["label"][-1]
            curr_pred_student_str = self.records["pred_student"][-1]
            
            # Compute the WER:
            self.records["wer_student"].append(100 * self.wer_metric.compute(references=[curr_label_str], predictions=[curr_pred_student_str]))  # type: ignore
            
            # Add boolean flag to indicate whether the prediction is correct:
            self.records["is_student_correct"].append(curr_label_str == curr_pred_student_str)
            
            # Add information about the current training state:
            self.records["epoch"].append(state.epoch)
            self.records["step"].append(state.global_step)
        
        
        # Log the records to wandb:
        # import pdb; pdb.set_trace()
        self.log_records_to_wandb()
        
        return
