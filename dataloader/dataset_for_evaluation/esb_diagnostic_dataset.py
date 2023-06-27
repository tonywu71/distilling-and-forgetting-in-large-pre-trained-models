from typing import Optional, List
from dataloader.dataset_for_evaluation.esb_dataset import ESBDataset


class ESBDiagnosticDataset(ESBDataset):
    """
    Class that regroups the diagnostic version of the End-to-end Speech Benchmark (ESB) datasets.
    See for more details:
    - https://arxiv.org/abs/2210.13352 
    - https://huggingface.co/datasets/esb/diagnostic-dataset
    """
    
    def __init__(self,
                 streaming: bool=False,
                 subset: Optional[List[str]]=None) -> None:
        
        super().__init__(streaming=streaming, load_diagnostic=True, subset=subset)
    