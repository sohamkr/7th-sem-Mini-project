# utils.py
import numpy as np
import joblib
from pathlib import Path

def save_numpy_array(path: str, arr: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def save_joblib(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
