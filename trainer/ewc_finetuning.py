from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.modeling_utils import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from utils.ewc_utils import load_ewc_params


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

        # Load the EWC parameters:
        self.ewc_mean_params, self.ewc_fisher_params = load_ewc_params(dirpath_ewc)
        

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
            loss += self.args.lambda_ewc * (self.args.ewc_fisher_params[name] * (param - self.args.ewc_mean_params[name]) ** 2).sum()
        
        return (loss, output_student) if return_outputs else loss
