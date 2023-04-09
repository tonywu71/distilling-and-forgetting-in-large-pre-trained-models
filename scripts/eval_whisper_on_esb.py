import pandas as pd
from tqdm.auto import tqdm

from transformers import pipeline
import evaluate
import typer

from dataloader.esb import ESB_Datasets
from normalization.whisper_normalization import get_whisper_normalizer


BATCH_SIZE = 16


def gen_from_dataset(dataset):
    """Yield the audio and reference from the dataset."""
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(model: str="openai/whisper-tiny.en"):
    """
    Evaluate the whisper model on the ESB benchmark.
    """
    esb_datasets = ESB_Datasets()
    
    whisper_asr = pipeline(
        task="automatic-speech-recognition",
        model=model
    )
    
    wer_metric = evaluate.load("wer")
    whisper_norm = get_whisper_normalizer(whisper_asr)
    
    def normalize_fct(batch):
        batch["norm_text"] = whisper_norm(esb_datasets.get_text(batch))
        return batch

    esb_datasets.preprocess_datasets(sampling_rate=whisper_asr.feature_extractor.sampling_rate,  # type: ignore
                                     normalize_fct=normalize_fct,
                                     n_samples=32)
    
    wer_results = []

    tbar = tqdm(esb_datasets.items())
    
    for dataset_name, dataset in tbar:
        tbar.set_description(f"Processing {dataset_name}")
        
        # Create placeholders for the predictions and references:
        predictions = []
        references = []
        
        for out in whisper_asr(gen_from_dataset(dataset), batch_size=BATCH_SIZE):  # type: ignore
            predictions.append(whisper_norm(out["text"]))  # type: ignore
            references.append(out["reference"][0])  # type: ignore
        
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, ndigits=3)  # type: ignore

        wer_results.append(wer)
    
    df = pd.DataFrame({"Dataset": esb_datasets.keys(), "WER": wer_results})
    print(df)
    
    return


if __name__ == "__main__":
    typer.run(main)
