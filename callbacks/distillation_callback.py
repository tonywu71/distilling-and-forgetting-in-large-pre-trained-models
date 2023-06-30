from typing import Dict, Optional, Tuple

import pandas as pd

import torch

from transformers import (PreTrainedModel,
                          WhisperProcessor,
                          TrainingArguments,
                          TrainerState,
                          TrainerControl)
from datasets import Dataset

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from evaluation.string_edit_metrics import get_string_edit_metrics
from callbacks.base_training_callback import BaseWandbTrainingCallback
from utils.distil_config import DistilConfig
from utils.constants import GEN_MAX_LENGTH, LOSS_MASK_IDX, DEFAULT_LABEL_TOKENIZED_COL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WandbDistillationCallback(BaseWandbTrainingCallback):
    """
    A callback for distillation that logs the first `n_samples` of the evaluation dataset to W&B.
    Also computes the WER of the student model's predictions (because `compute_metrics` is not
    called during distillation).
    """
    
    def __init__(self,
                 config: DistilConfig,
                 processor: WhisperProcessor,
                 eval_dataset: Dataset,
                 n_samples: int,
                 teacher_model: Optional[PreTrainedModel] = None,
                 log_raw_str: bool = False):
        super().__init__(config,
                         processor,
                         eval_dataset,
                         n_samples,
                         log_raw_str)
        
        assert isinstance(self.config, DistilConfig), "config must be `DistilConfig`"
        
        self.table_name = "sample_predictions-distill"
        self.data_collator_no_k_beam = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=self.processor.tokenizer,
                                                                            feature_extractor=self.processor.feature_extractor,
                                                                            add_k_beam_features=False)
        
        self.is_seq_level = (self.config.method_distil in ["seq_level_uniform", "seq_level_k_best_ranked"])
        
        if self.is_seq_level:  # If sequence-level distillation...
            self.data_collator_with_k_beam = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=self.processor.tokenizer,
                                                                                  feature_extractor=self.processor.feature_extractor,
                                                                                  add_k_beam_features=True)
        else:  # If word-level distillation...
            assert teacher_model is not None, "`teacher_model` must be provided for word-level distillation"
            self.teacher_model = teacher_model
    
    
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
        # NOTE: `model` corresponds to the student model in this method.
        
        # Call `BaseWandbTrainingCallback`'s parent (`WandbCallback`) method `on_log` for basic logging:
        super(BaseWandbTrainingCallback, self).on_log(args, state, control, model, logs, **kwargs)  # type: ignore
        
        # Log the predictions of the student model:
        self.predict_and_log_preds_to_wandb(model, state)
        
        return

    
    def predict_and_log_preds_to_wandb(self,
                                       model: PreTrainedModel,
                                       state: TrainerState) -> None:
        # Iterate through the first n samples:
        for idx, data in enumerate(self.eval_dataset):
            if idx >= self.n_samples:
                break
            
            # Log the original audio (should be done before call to DataCollator):
            self.log_audio_to_records(data)
            
            # Get labels and predictions:
            label_ids, pred_ids_teacher, pred_ids_student = self.get_labels_and_preds(student_model=model, data=data)
            
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
            curr_pred_teacher_str = self.records["pred_teacher"][-1]
            curr_pred_student_str = self.records["pred_student"][-1]
            
            # Compute the WER of both the teacher and the student:
            self.records["wer_teacher"].append(100 * get_string_edit_metrics(references=[curr_label_str], predictions=[curr_pred_teacher_str])["wer"])  # type: ignore
            self.records["wer_student"].append(100 * get_string_edit_metrics(references=[curr_label_str], predictions=[curr_pred_student_str])["wer"])  # type: ignore
            
            # Add boolean flag to indicate whether the prediction is correct:
            self.records["is_student_correct_wrt_teacher"].append(curr_pred_student_str == curr_pred_teacher_str)
            self.records["is_student_correct_wrt_label"].append(curr_pred_student_str == curr_label_str)
            
            # Add information about the current training state:
            self.records["epoch"].append(state.epoch)
            self.records["step"].append(state.global_step)
        
        
        # Log the records to wandb:
        self.log_records_to_wandb()
        return


    def get_labels_and_preds(self,
                             student_model: PreTrainedModel,
                             data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the labels and the teacher and student predictions. Automatically handles both word-level and
        sequence-level distillation.
        """
        if self.is_seq_level:  # If sequence-level distillation...
            # Collate the data into batches of size 1:
            data = self.data_collator_with_k_beam([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device)
            label_ids = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==LOSS_MASK_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Generate the predictions:
            pred_ids_teacher = data["teacher_sequences"][0:1, 0, :]  # get 1st element of the size-1 batch and 1st beam
            pred_ids_student = student_model.generate(input_features,
                                                      max_length=GEN_MAX_LENGTH,
                                                      num_beams=self.config.generation_num_beams)  # type: ignore
        
        else:  # If word-level distillation...
            # Collate the data into batches of size 1:
            # NOTE: `data_collator_no_k_beam` because we haven't loaded the dataset with k-beam
            data = self.data_collator_no_k_beam([data])  # type: ignore
            
            # Note that we need to move the data to the device manually (which is not the case with Trainer):
            input_features = data["input_features"].to(device)
            label_ids = data[DEFAULT_LABEL_TOKENIZED_COL].to(device)
            
            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly during decoding:
            label_ids[label_ids==LOSS_MASK_IDX] = self.processor.tokenizer.pad_token_id  # type: ignore
            
            # Generate the predictions:
            pred_ids_student = student_model.generate(input_features,
                                                      max_length=GEN_MAX_LENGTH,
                                                      num_beams=self.config.generation_num_beams)  # type: ignore
            pred_ids_teacher = self.teacher_model.generate(input_features,
                                                           max_length=GEN_MAX_LENGTH,
                                                           num_beams=self.config.generation_num_beams)  # type: ignore
        
        return label_ids, pred_ids_teacher, pred_ids_student
