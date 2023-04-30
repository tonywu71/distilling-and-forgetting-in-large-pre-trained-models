from collections import defaultdict
from typing import Dict, Optional
import itertools

import pandas as pd

from torch.utils.data import DataLoader

from transformers import GenerationMixin, PreTrainedTokenizer, GenerationMixin
from transformers.integrations import WandbCallback
import evaluate

from utils.config import Config
from utils.constants import GEN_MAX_LENGTH, PADDING_IDX


class WandbCustomCallback(WandbCallback):
    """
    Custom Huggingface's Trainer callback to log the progress results to W&B.
    """
    
    def __init__(self, config: Config, n_batches: int=1):
        super().__init__()
        self.config = config
        self.n_batches = n_batches
        self.wer_metric = evaluate.load("wer")
    
    
    def on_log(self,
               args,
               state,
               control,
               model: GenerationMixin,
               tokenizer: PreTrainedTokenizer,
               eval_dataloader: DataLoader,
               logs: Optional[Dict[str, float]]=None,
               **kwargs):
        
        super().on_log(args, state, control, model, logs, **kwargs)
        
        # Slice the dataloader to get the first n batches
        sliced_dataset = itertools.islice(eval_dataloader, self.n_batches)
        
        records = defaultdict(list)
        
        # Iterate through the first n samples
        for i, data in enumerate(sliced_dataset):
            inputs, label_ids = data
        
            pred_ids = model.generate(inputs,
                                      max_length=GEN_MAX_LENGTH,
                                      num_beams=self.config.generation_num_beams)

            # Replace the padding index with the pad token id to undo the step we applied
            # in the data collator to ignore padded tokens correctly in the loss:
            label_ids[label_ids==PADDING_IDX] = tokenizer.pad_token_id
            
            # Decode the predictions:
            batch_pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)
            
            # Decode the labels:
            batch_label_str_raw = tokenizer.batch_decode(label_ids, skip_special_tokens=False, normalize=False)
            batch_label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=True)

            for pred_str, label_str_raw, label_str in zip(batch_pred_str, batch_label_str_raw, batch_label_str):
                records["pred_str"].append(pred_str)
                records["label_str_raw"].append(label_str_raw)
                records["label_str"].append(label_str)
                
                # Compute the WER in percent per example:
                records["wer"].append(100 * self.wer_metric.compute(references=label_str, predictions=pred_str))  # type: ignore
        
        # Create a new wandb table:
        table = self._wandb.Table(dataframe=pd.DataFrame(records))
        self._wandb.log({"sample_predictions": table})
        
        return
