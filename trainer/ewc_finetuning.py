from pathlib import Path

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.modeling_utils import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from safetensors import safe_open


class EWCFinetuningTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments for fine-tuning with Elastic Weight Consolidation (EWC).
    Should be used with `EWCFinetuningTrainer`.
    """
    
    def __init__(self,
                 dirpath_ewc: str,
                 lambda_ewc: float,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dirpath_ewc = dirpath_ewc
        self.lambda_ewc = lambda_ewc
        
        self.dirpath_ewc_mean = Path(self.dirpath_ewc) / "ewc_mean_params.safetensors"
        assert self.dirpath_ewc_mean.exists(), f"`dirpath_ewc_mean` does not exist: {self.dirpath_ewc_mean}"
        
        self.dirpath_ewc_fisher = Path(self.dirpath_ewc) / "ewc_fisher_params.safetensors"
        assert self.dirpath_ewc_fisher.exists(), f"`dirpath_ewc_fisher` does not exist: {self.dirpath_ewc_fisher}"
        
        self.ewc_mean_params = {}
        with safe_open(self.dirpath_ewc_mean, framework="pt", device=0) as f:
            for k in f.keys():
                self.ewc_mean_params[k] = f.get_tensor(k)
        
        self.ewc_fisher_params = {}
        with safe_open(self.dirpath_ewc_fisher, framework="pt", device=0) as f:
            for k in f.keys():
                self.ewc_fisher_params[k] = f.get_tensor(k)
        

class EWCFinetuningTrainer(Seq2SeqTrainer):
    """
    Trainer class for fine-tuning with Elastic Weight Consolidation (EWC).
    Should be used with `args=EWCFinetuningTrainingArguments`.
    """
    
    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs: BatchFeature,
                     return_outputs: bool = False):
        """
        Override the `compute_loss` method from `Seq2SeqTrainer`.
        """
        loss, output_student = super().compute_loss(model, inputs, return_outputs=True)
        
        # Compute and add the EWC penalty term for each weight tensor:
        for name, param in model.named_parameters():
            loss += self.args.lamda_ewc * self.args.ewc_fisher_params * (param - self.args.ewc_mean_params[name]).pow(2).sum()
        
        return (loss, output_student) if return_outputs else loss
