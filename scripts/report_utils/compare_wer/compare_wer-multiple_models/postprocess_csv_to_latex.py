import typer
import pandas as pd


def main(filepath: str):
    """
    Script that takes a CSV output from `compare_multiple_models_to_latex.py` and
    postprocess it to make it easier to read.
    
    Used to generate the table in the "Impact of over-training on forgetting" section
    of the April Report.
    """
    
    df = pd.read_csv(filepath).set_index("Dataset")
    
    df.index.name = "Checkpoints"
    df = df.T
    df.index = df.index.str.extract(r'checkpoint-(\d+)-fab').astype(int).values.flatten()  # type: ignore
    
    output = df.to_latex()
    
    print("```latex")
    print(output + "```")
    
    return


if __name__ == "__main__":
    typer.run(main)
