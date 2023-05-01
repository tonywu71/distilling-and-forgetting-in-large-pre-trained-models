from pathlib import Path
import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm

from transformers import (pipeline,
                          WhisperForConditionalGeneration,
                          WhisperProcessor)
import evaluate

import wandb

from dataloader.dataloader import gen_from_dataset
from dataloader.esb import ESB_Datasets
from normalization.whisper_normalization import get_whisper_normalizer

from utils.constants import DEFAULT_LABEL_STR_COL, DEFAULT_OUTPUT_DIR, WANDB_PROJECT


def extract_model_name_from_path(filepath: str) -> str:
    """Extract the model name from a path."""
    path = Path(filepath)
    
    if "checkpoint-" in path.parts:
        filename = f"{path.parent.stem}-{path.stem}.csv"
    else:
        filename = f"{path.stem}.csv"
    
    return filename


def main(pretrained_model_name_or_path: str,
         subset: Optional[List[str]]=typer.Option(None, help="Subset of the ESB benchmark to evaluate on."),
         n_samples: int=typer.Option(128, help="Number of samples to evaluate on."),
         batch_size: int=typer.Option(16, help="Batch size for the ASR pipeline."),
         filename: Optional[str]=typer.Option(
             None, help="Filename of the output CSV file. Leave to `None` to use the stem of `pretrained_model_name_or_path` as the file name.")) -> None:
    """
    Evaluate the whisper model on the ESB benchmark.
    Note that only greedy decoding is supported for now.
    """
    
    # -----------------------   W&B   -----------------------
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "subset": subset,
        "n_samples": n_samples,
        "batch_size": batch_size
    }
    
    wandb.login()
    wandb.init(project=WANDB_PROJECT,
               job_type="eval_esb",
               name=f"eval_esb-{Path(pretrained_model_name_or_path).stem}",
               config=config)
    
    # Sanity checks:
    assert batch_size <= n_samples, "Batch size must be smaller than the number of samples."
    if subset:
        print(f"Subset(s) of ESB: {subset}")
    
    # Load dataset:
    esb_datasets = ESB_Datasets(subset=subset)
    print(f"Loaded datasets: {list(esb_datasets.keys())}")
    
    
    # Load pipeline:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path)
    whisper_asr = pipeline(task="automatic-speech-recognition",
                           model=model,
                           tokenizer=processor.tokenizer,  # type: ignore
                           feature_extractor=processor.feature_extractor,  # type: ignore
                           device=0  # use 1st GPU for Whisper
    )
    
    
    # Preprocess the datasets:
    whisper_norm = get_whisper_normalizer(whisper_asr.tokenizer)  # type: ignore
    
    def normalize_fct(batch):
        batch[DEFAULT_LABEL_STR_COL] = whisper_norm(esb_datasets.get_text(batch))
        return batch

    esb_datasets.preprocess_datasets(sampling_rate=whisper_asr.feature_extractor.sampling_rate,  # type: ignore
                                     normalize_fct=normalize_fct,
                                     n_samples=n_samples)
    
    
    # Load metric:
    wer_metric = evaluate.load("wer")
    
    
    # Loop over the datasets:
    wer_results = []
    tbar = tqdm(esb_datasets.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}...")
        
        # Create placeholders for the predictions and references:
        predictions = []
        references = []
        
        for out in whisper_asr(gen_from_dataset(dataset), batch_size=batch_size):  # type: ignore
            predictions.append(whisper_norm(out["text"]))  # type: ignore
            references.append(out["reference"][0])  # type: ignore
        
        assert predictions, "Empty batch. Try to run with a higher value of `n_samples`."
        
        # Compute the WER in percent:
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, ndigits=3)  # type: ignore

        wer_results.append(wer)
    
    
    # Save the results:
    results = pd.Series(wer_results, index=list(esb_datasets.keys()), name="WER (%)")
    results.index.name = "Dataset"
    
    # Compute the average:
    results["Average"] = results.mean()
    
    print("Results:")
    print(results)
    print()
    
    if filename is None:
        filename = extract_model_name_from_path(pretrained_model_name_or_path) + ".csv"
    
    filepath = DEFAULT_OUTPUT_DIR / filename
    filepath.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(f"{filepath}")
    print(f"Results saved to `{filepath}`.")
    
    barplot = wandb.plot.bar(wandb.Table(dataframe=results.to_frame().reset_index()),  # type: ignore
                             label=results.index.name,
                             value=str(results.name),
                             title="Per dataset WER (%)")
    wandb.log({"wer_for_esb_dataset": barplot})
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
