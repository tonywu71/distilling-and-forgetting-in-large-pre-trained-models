from typing import Any, Optional, Literal, Dict
from abc import ABC

import torch
from torch.utils.data import DataLoader, Dataset

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.whisper import WhisperProcessor

from dataloader.collator import DataCollatorSpeechSeq2SeqWithPadding
from dataloader.utils import get_fast_tokenizer_from_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainingArgumentsBase(ABC, Seq2SeqTrainingArguments):
    """
    Subclass of `TrainingArguments` used for `DistillationTrainer`.
    Only supports distillation for non-sequential tasks.
    """
    def __init__(self,
                 method_distil: Literal["word_level", "seq_level_uniform", "seq_level_ranked"],
                 *args,
                 **kwargs):
        kwargs = self.prepare_kwargs(**kwargs)
        super().__init__(*args, **kwargs)
        self.method_distil = method_distil
    
    def prepare_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare the kwargs for the `__init__` method of `DistillationTrainingArgumentsBase`.
        """
        list_keys_to_remove = [
            "method_distil",
            "alpha_ce",
            "temperature",
            "distillation_num_beams",
            "beta_decay"
        ]
        for key in list_keys_to_remove:
            kwargs.pop(key, None)
        return kwargs


class DistillationTrainerBase(ABC, Seq2SeqTrainer):
    """
    Trainer class for distillation. Should be used with `args=DistillationTrainingArguments`.
    """
    def __init__(self,
                 args: DistillationTrainingArgumentsBase,
                 student_processor: WhisperProcessor,
                 **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.student_processor = student_processor
        self.student_tokenizer = get_fast_tokenizer_from_tokenizer(self.student_processor.tokenizer)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        """
        Overriding `get_eval_dataloader` as we should not pass the attention mask during evaluation.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer=self.student_tokenizer,
                                                             feature_extractor=self.student_processor.feature_extractor,
                                                             return_attention_mask=False)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
