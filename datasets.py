
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, horizon: int = 1):
        """
        data
        """
        X, y = [], []
        T = data.shape[0]
        for i in range(seq_len, T - horizon + 1):
            X.append(data[i-seq_len:i, :-1])
            y.append(data[i + horizon - 1, -1])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
