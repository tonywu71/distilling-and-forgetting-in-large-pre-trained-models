from typing import Dict

import torch
from transformers.models.whisper import WhisperTokenizer


class TrainerDebugger:
    def __init__(self,
                 tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer


    def inspect_batch(self,
                      inputs: Dict[str, torch.Tensor],
                      idx_sample: int = 0):
        """
        Inspect a batch of inputs.
        
        Inputs:
        - `inputs` has the following keys: `input_ids`, `labels`, `attention_mask`
        - `idx_sample` is the index of the sample to inspect.
        """
        
        print("=========   Batch-wise   =========")
        print(f"`input_features` shape: {inputs['input_features'].shape}")
        print(f"`labels` shape: {inputs['labels'].shape}")
        print(f"`attention_mask` shape: {inputs['attention_mask'].shape}")
        print()
        
        
        
        print(f"=========   Sample-wise (sample {idx_sample})   =========")
        
        sample = {key: inputs[key][idx_sample] for key in inputs}
        
        print("`input_features`:")
        print(sample["input_features"])
        print(f"`input_features` shape: {sample['input_features'].shape}")
        
        print()
        
        print("`labels`:")
        print(sample["labels"])
        print(f"`labels` shape: {sample['labels'].shape}")
        
        print()
        
        decoded_labels = self.tokenizer.batch_decode([sample['labels']], skip_special_tokens=False, normalize=False)
        print(f"`labels` decoded: {decoded_labels}")
        
        print()
        
        print("`attention_mask`:")
        print(sample["attention_mask"])
        print(f"`attention_mask` shape: {sample['attention_mask'].shape}")
        
        print()
        
        return
    