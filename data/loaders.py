# data/loaders.py
from typing import Tuple
import pandas as pd
import numpy as np

def load_csv(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df

def time_train_test_split(df: pd.DataFrame, timestamp_col="timestamp", test_size_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_ts = df[timestamp_col].max()
    cutoff = last_ts - pd.Timedelta(days=test_size_days)
    train = df[df[timestamp_col] <= cutoff].reset_index(drop=True)
    test = df[df[timestamp_col] > cutoff].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        split_i = int(len(df) * 0.8)
        train, test = df.iloc[:split_i].copy().reset_index(drop=True), df.iloc[split_i:].copy().reset_index(drop=True)
    return train, test
