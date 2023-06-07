import pandas as pd

def post_process_esb_librispeech(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["librispeech_clean", "librispeech_other"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["In-domain average", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Out-of-domain average", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df


def post_process_esb_ami(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["ami"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["In-domain average", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Out-of-domain average", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df


def post_process_mls(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IN_DOMAIN = ["english"]
    COLS_TO_EXCLUDE = ["Average"]
    
    df = df.copy()
    
    cols_out_of_domain = df.index.difference(COLS_IN_DOMAIN).difference(COLS_TO_EXCLUDE)

    df.loc["English", :] = df.loc[COLS_IN_DOMAIN, :].mean()
    df.loc["Non-English languages", :] = df.loc[cols_out_of_domain, :].mean()
    
    return df
