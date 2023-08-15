import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.initialize import initialize_env
initialize_env()

from typing import List, Optional
from pathlib import Path
from pprint import pprint 

from transformers import set_seed
import wandb

from dataloader.dataset_for_evaluation.base_dataset_group import BaseDatasetGroup
from evaluation.eval_dataset_name_to_dataset_group import EVAL_DATASET_NAME_TO_DATASET_GROUP
from evaluation.eval_whisper_implicit_lm_on_dataset import eval_whisper_implicit_lm_on_dataset_group
from evaluation.eval_whisper_on_dataset_group import eval_whisper_wer_on_dataset_group
from evaluation.eval_whisper_utils import log_wer_to_wandb, save_edit_metrics_to_csv, log_edit_metrics_to_wandb
from utils.file_io import extract_exp_name_from_model_path, extract_output_savepath_from_model_path
from utils.constants import DEFAULT_EVAL_BATCH_SIZE, DEFAULT_EVAL_NUM_BEAMS



def main(pretrained_model_name_or_path: str = typer.Argument(..., help="Path to the pretrained model."),
         dataset_name: str = typer.Option(..., help="Name of the dataset to evaluate on."),
         streaming: bool = typer.Option(False, help="Whether to use streaming inference."),
         subset: Optional[List[str]] = typer.Option(None, help="Subset of the ESB dataset to evaluate on."),
         filter_audio_length: bool = typer.Option(False, help="Whether to filter out audio files that are too short or too long. Disabled by default."),
         task: str = typer.Option("transcribe", help="Task to evaluate on."),
         zero_shot: bool = typer.Option(False, help="Whether to use zero-shot inference. Defaults to False."),
         num_beams: int = typer.Option(DEFAULT_EVAL_NUM_BEAMS, help="Number of beams for decoding."),
         batch_size: int = typer.Option(DEFAULT_EVAL_BATCH_SIZE, help="Batch size for decoding."),
         no_repeat_ngram_size: Optional[int] = typer.Option(None, help="No repeat ngram size for decoding."),
         sampling: bool = typer.Option(False, help="Whether to use sampling for decoding."),
         gen_top_k: Optional[int] = typer.Option(None, help="Top-k for decoding."),
         gen_temperature: float = typer.Option(1., help="Temperature for decoding."),
         gen_top_p: Optional[float] = typer.Option(None, help="Top-p for decoding."),
         seed: Optional[int] = typer.Option(None, help="Set seed to reproduce results."),
         savepath: Optional[str] = typer.Option(None, help="Path of the output CSV file. Leave to `None` to use the name of `pretrained_model_name_or_path` as the filename."),
         implicit_lm_ppl: bool = typer.Option(False, help="Whether to evaluate the implicit language model perplexity or not."),
         save_preds: bool = typer.Option(False, help="Whether to save the predictions in a JSON file or not."),
         debug: bool = typer.Option(False, help="Whether to run in debug mode or not.")) -> None:
    """
    Evaluate the pre-trained Whisper model on a DatasetGroup instance.
    """
    
    assert dataset_name in EVAL_DATASET_NAME_TO_DATASET_GROUP.keys(), f"Dataset name must be one of {list(EVAL_DATASET_NAME_TO_DATASET_GROUP.keys())}."
    if num_beams > 1:
        assert (gen_top_k is None) and (gen_top_p is None), "Cannot use both `num_beams` and `gen_top_k`/`gen_top_p`."
    
    if seed:
        print(f"Setting seed to {seed}...")
        set_seed(seed)
    
    # Prepare generate_kwargs:
    generate_kwargs = {}
    if no_repeat_ngram_size:
        generate_kwargs.update({"no_repeat_ngram_size": no_repeat_ngram_size})
    if sampling:
        generate_kwargs.update({"do_sample": True, "top_k": 0, "temperature": gen_temperature})  # override the default value of `top_k` which is 50
    if gen_top_k:
        generate_kwargs.update({"top_k": gen_top_k})
    if gen_top_p:
        generate_kwargs.update({"top_p": gen_top_p})
    
    if generate_kwargs:
        print("\n-----------------------\n")
        pprint(generate_kwargs)
        print("\n-----------------------\n")
    
    # Load dataset:
    dataset_group: BaseDatasetGroup = EVAL_DATASET_NAME_TO_DATASET_GROUP[dataset_name](streaming=streaming, subset=subset)
    
    # Create config for wandb:
    config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "dataset_name": dataset_name,
        "language": dataset_group.language,
        "streaming": streaming,
        "subset": subset,
        "task": task,
        "zero_shot": zero_shot,
        "num_beams": num_beams,
        "batch_size": batch_size,
        "savepath": savepath,
        "eval_implicit_lm_ppl": implicit_lm_ppl,
        "save_preds": save_preds
    }

    # Initialize W&B:
    wandb.login()

    tags = [dataset_name]
    name = f"eval_{dataset_name}-{extract_exp_name_from_model_path(pretrained_model_name_or_path)}"
    if implicit_lm_ppl:
        tags.append("implicit_lm_ppl")
        name += "-implicit_lm_ppl"

    wandb.init(project=os.environ["WANDB_PROJECT_EVALUATION"],
               job_type="evaluation",
               tags=tags,
               name=name,
               config=config,
               mode="disabled" if debug else None)
    
    # Print config:
    print("Parameters:")
    pprint(config)
    print("\n-----------------------\n")
    
    # Load dataset:
    if subset:
        print(f"Subset(s) of {dataset_name}: {subset}")
    
    # Print loaded datasets:
    print(f"Loaded datasets: {list(dataset_group.keys())}")
    
    # If needed, filter out audio files that are too short or too long:
    if filter_audio_length:
        print("Filtering out audio files that are too short or too long...")
        dataset_group.filter_audio_length(verbose=True)
    
    # Evaluate:
    print("Evaluating...")

    if implicit_lm_ppl:
        results = eval_whisper_implicit_lm_on_dataset_group(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                            ds_group=dataset_group,
                                                            batch_size=batch_size,
                                                            task=task,
                                                            zero_shot=zero_shot)

        print("Results:")
        print(results)

        # Save results:
        if savepath is None:
            savepath = extract_output_savepath_from_model_path(pretrained_model_name_or_path) + "-implicit_lm-ppl" + f"-{dataset_name}"
            if zero_shot:
                savepath += "-zero_shot"
            savepath += ".csv"
        
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        results.to_csv(f"{savepath}")
        print(f"Results saved to `{savepath}`.")
        
        # Log results to W&B:
        barplot = wandb.plot.bar(wandb.Table(dataframe=results.to_frame().reset_index()),  # type: ignore
                                label=results.index.name,
                                value=str(results.name),
                                title="Per dataset perplexity (%)")
        wandb.log({"perplexity_for_dataset_group": barplot})
        wandb.finish()

    else:
        df_edit_metrics = eval_whisper_wer_on_dataset_group(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                            ds_group=dataset_group,
                                                            task=task,
                                                            zero_shot=zero_shot,
                                                            batch_size=batch_size,
                                                            num_beams=num_beams,
                                                            save_preds=save_preds,
                                                            generate_kwargs=generate_kwargs)
        
        # Round the results:
        df_edit_metrics = df_edit_metrics.round(2)
        
        # Split the results into two dataframes:
        df_edit_metrics_ortho = df_edit_metrics[["WER ortho (%)", "Sub ortho (%)", "Del ortho (%)", "Ins ortho (%)"]]
        df_edit_metrics_norm = df_edit_metrics[["WER (%)", "Sub (%)", "Del (%)", "Ins (%)"]]
        
        
        print("\n-----------------------\n")
        
        print("Orthometric results:")
        print(df_edit_metrics_ortho)
        
        print("\n-----------------------\n")
        
        print("Normalized results:")
        print(df_edit_metrics_norm)
        
        print("\n-----------------------\n")
        
        
        # Save and log the edit metrics:
        for df, suffix in zip([df_edit_metrics_ortho, df_edit_metrics_norm], ["orthographic", "normalized"]):
            save_edit_metrics_to_csv(df_edit_metrics=df,
                                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                                    dataset_name=dataset_name,
                                    savepath=savepath,
                                    suffix=suffix)
            log_edit_metrics_to_wandb(df_edit_metrics=df, suffix=suffix)
            if suffix == "normalized":
                log_wer_to_wandb(wer_metrics=df["WER (%)"], suffix=suffix)
            else:
                log_wer_to_wandb(wer_metrics=df["WER ortho (%)"], suffix=suffix)
    
    wandb.finish()
    
    return


if __name__ == "__main__":
    typer.run(main)
