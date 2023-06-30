import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import torch

from utils.initialize import initialize_env
initialize_env()

from pprint import pprint

import pandas as pd
from transformers import (pipeline,
                          WhisperProcessor,
                          WhisperForConditionalGeneration)
from datasets import load_dataset

import wandb

from benchmark.speed_benchmark import get_speed_benchmark
from utils.file_io import extract_exp_name_from_model_path, extract_output_savepath_from_model_path


def main(pretrained_model_name_or_path: str = typer.Argument(..., help="Path to pretrained model.")):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    LIBRISPEECH_DUMMY_PATH = "hf-internal-testing/librispeech_asr_dummy"
    LANGUAGE = "english"
    TASK = "transcribe"
    NUM_BEAMS = 5
    NUM_WARMUP = 10
    NUM_TIMED_RUNS = 100
    
    # Create config for wandb:
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "language": LANGUAGE,
        "task": TASK,
        "dataset_name": LIBRISPEECH_DUMMY_PATH,
        "query_label": None,
        "DEVICE": device,
        "num_beams": NUM_BEAMS,
        "num_warmup": NUM_WARMUP,
        "num_timed_runs": NUM_TIMED_RUNS
    }
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_EVALUATION"],
               job_type="speed-benchmark",
               name=f"speed_benchmark-{extract_exp_name_from_model_path(pretrained_model_name_or_path)}",
               config=config)
    
    # Load dataset:
    cache_dir_librispeech = os.environ.get("CACHE_DIR_LIBRISPEECH", None)
    if cache_dir_librispeech is None:
        print("WARNING: `CACHE_DIR_LIBRISPEECH` environment variable not set. Using default cache directory.")
    else:
        print(f"Using cache directory: `{cache_dir_librispeech}`.")
    dataset = load_dataset(LIBRISPEECH_DUMMY_PATH, "clean", split="validation", cache_dir=cache_dir_librispeech)
    
    # Get reference query from the first example of the dataset:
    sample = dataset[0]
    query = {"raw": sample["audio"]["array"], "sampling_rate": sample["audio"]["sampling_rate"]}
    label = sample["text"]
    
    # Add reference query label to config:
    config["query_label"] = label
    
    # Print config:
    print("Parameters:")
    pprint(config)
    
    
    print("\n-----------------------\n")
    print("⚠️ Make sure that all speed benchmarks are run with on the same hardware to get consistent results.")
    print("\n-----------------------\n")
    
    
    # Load model:
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path,
                                                 language=LANGUAGE,
                                                 task=TASK)
        
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)  # type: ignore
    
    # Create pipeline:
    whisper_asr = pipeline(task="automatic-speech-recognition",
                            model=model,  # type: ignore
                            tokenizer=processor.tokenizer,  # type: ignore
                            feature_extractor=processor.feature_extractor,  # type: ignore
                            device=device
    )
    
    # Get speed benchmark:
    time_avg_ms, time_std_ms = get_speed_benchmark(pipeline=whisper_asr,
                                                   query=query,
                                                   num_warmup=NUM_WARMUP,
                                                   num_timed_runs=NUM_TIMED_RUNS,
                                                   num_beams=NUM_BEAMS)
    
    print(f"Average latency (ms): {time_avg_ms:.2f} ± {time_std_ms:.2f}")
    
    # Create Series:
    ser_speed_benchmark = pd.Series({"Average latency (ms)": time_avg_ms,
                                     "Standard deviation (ms)": time_std_ms},
                                    name=extract_exp_name_from_model_path(pretrained_model_name_or_path))
    
    # Save results:
    savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + "-speed_benchmark.csv"
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    ser_speed_benchmark.to_csv(savepath)
    print(f"Speed benchmark saved to `{savepath}`.")

    # Log summary results to W&B:
    wandb.run.summary["Average latency (ms)"] = time_avg_ms
    wandb.run.summary["Standard deviation latency (ms)"] = time_std_ms
    
    return


if __name__ == "__main__":
    typer.run(main)
