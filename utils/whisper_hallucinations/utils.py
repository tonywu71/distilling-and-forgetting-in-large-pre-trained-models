import pandas as pd

def get_iqr(df: pd.DataFrame, col: str) -> float:
    assert col in df.columns, f"Column '{col}' not in DataFrame."
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return iqr
