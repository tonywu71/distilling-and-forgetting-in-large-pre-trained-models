import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from safetensors.torch import save_file

from utils.ewc_utils import load_ewc_params


def main(dirpath_1: str = typer.Argument(..., help="The directory path of the first EWC params."),
         dirpath_2: str = typer.Argument(..., help="The directory path of the second EWC params."),
         savepath: str = typer.Argument(..., help="The directory path to save the combined EWC params."),
         alpha_1: float = typer.Option(0.5, help="The weight for the first EWC params.")):
    mean_params_1, fisher_params_1 = load_ewc_params(dirpath_1)
    mean_params_2, fisher_params_2 = load_ewc_params(dirpath_2)
    
    mean_params = {}
    fisher_params = {}
    for param_name in mean_params_1:
        mean_params[param_name] = alpha_1 * mean_params_1[param_name] + (1 - alpha_1) * mean_params_2[param_name]
        fisher_params[param_name] = alpha_1 * fisher_params_1[param_name] + (1 - alpha_1) * fisher_params_2[param_name]
    
    # Save the EWC params:
    Path(savepath).mkdir(parents=True, exist_ok=True)
    
    mean_params_savepath = os.path.join(savepath, "ewc_mean_params.safetensors")
    save_file(mean_params, mean_params_savepath)
    print(f"Saved the EWC mean parameters to `{mean_params_savepath}`.")
    
    fisher_params_savepath = os.path.join(savepath, "ewc_fisher_params.safetensors")
    save_file(fisher_params, fisher_params_savepath)
    print(f"Saved the EWC Fisher parameters to `{fisher_params_savepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
