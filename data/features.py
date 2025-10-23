# data/features.py
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def create_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for lag in lags:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    out = out.dropna().reset_index(drop=True)
    return out

def add_rolling_features(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_roll_mean_{w}"] = out[c].rolling(window=w, min_periods=1).mean()
            out[f"{c}_roll_std_{w}"] = out[c].rolling(window=w, min_periods=1).std().fillna(0)
    return out

def fit_scaler(arr: np.ndarray, save_path: str = None) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(arr)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
    return scaler

def load_scaler(path: str):
    return joblib.load(path)
