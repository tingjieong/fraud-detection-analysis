from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
RAW = Path(__file__).resolve().parents[1] / "datasets" / "creditcard.csv"


def load_raw(path=RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def dedupe(df: pd.DataFrame):
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def temporal_split(df, train_frac=0.70, calib_frac=0.10, time_col="Time"):
    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_tr = int(n * train_frac)
    i_ca = int(n * (train_frac + calib_frac))
    return (df.iloc[:i_tr].copy(),
            df.iloc[i_tr:i_ca].copy(),
            df.iloc[i_ca:].copy())


def random_split(df, test_size=0.30, seed=RANDOM_STATE):
    return train_test_split(df, test_size=test_size,
                            stratify=df["Class"], random_state=seed)