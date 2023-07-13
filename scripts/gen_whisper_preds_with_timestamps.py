import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from tqdm.auto import tqdm

import torch
import whisper
from datasets import Dataset

import wandb

from dataloader.dataset_loader import TRAIN_DATASET_NAME_TO_LOAD_FCT
from evaluation.eval_dataset_name_to_dataset_group import EVAL_DATASET_NAME_TO_DATASET_GROUP
from evaluation.eval_whisper_utils import save_preds_to_json
from utils.constants import DEFAULT_OUTPUT_DIR, GEN_MAX_LENGTH


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name in ["ami_100h_train", "ami_10h_train"]:
        dataset_dict = TRAIN_DATASET_NAME_TO_LOAD_FCT[dataset_name]()
        ds = dataset_dict["train"]
    elif dataset_name in ["librispeech_dummy", "ami_test", "ami_10h_test"]:
        dataset_name = dataset_name.replace("_test", "")
        ds_group = EVAL_DATASET_NAME_TO_DATASET_GROUP[dataset_name]()
        if dataset_name == "librispeech_dummy":
            ds = ds_group.str2dataset["librispeech_dummy"]
            ds = ds.map(lambda x: {"text": x.lower()}, input_columns=["text"])
        elif dataset_name in ["ami", "ami_10h"]:
            ds = ds_group.str2dataset["ami"]
            ds = ds.map(lambda x: {"text": x.lower()}, input_columns=["text"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")
    return ds


def main(model_name: str = typer.Argument(..., help="The name of the model to use. Naming conventions are defined in the official Whisper repository."),
         dataset_name: str = typer.Argument(..., help="The name of the (eval) dataset to use."),
         debug: bool = typer.Option(False, help="Whether to run in debug mode or not.")) -> None:
    
    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_OTHERS"],
               job_type="gen_whisper_preds_with_timestamps",
               tags=[model_name, dataset_name],
               name=f"gen_whisper_preds_with_timestamps_{model_name}_{dataset_name}",
               mode="disabled" if debug else None)
    
    # Get the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Whisper model
    model = whisper.load_model(model_name, device=device)
    print(f"Loaded Whisper model `{model_name}`.")

    # Load the dataset:
    ds = load_dataset(dataset_name).with_format("torch")
    print(f"Loaded dataset `{dataset_name}`.")

    # Create placeholders:
    results = []
    references = []

    # Predict:
    print("Predicting...")
    for sample in tqdm(ds, total=ds.num_rows):
        audio = sample["audio"]["array"]
        if device == "cuda":
            audio = audio.to(torch.float16)
        else:
            pass  # keep the default float32
        results.append(model.transcribe(audio,
                                        language="en",
                                        temperature=0.0,
                                        word_timestamps=True,
                                        beam_size=1,
                                        max_length=GEN_MAX_LENGTH))
        references.append(sample["text"].lower())
    
    # Save the results:
    savepath = f"outputs/whisper_preds_with_timestamps/{model_name}/{dataset_name}.json"
    savepath = (DEFAULT_OUTPUT_DIR / "whisper_preds_with_timestamps" / model_name / dataset_name).with_suffix(".json")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    save_preds_to_json(references=references, predictions=results, savepath=savepath)
    print(f"Saved predictions to `{savepath}`.")

    wandb.finish()

    return


if __name__ == "__main__":
    typer.run(main)
