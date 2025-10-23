# trainers/seq_trainer.py
from typing import Optional, Dict
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

def train_seq_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader]=None,
                    epochs: int = 50, lr: float = 1e-3, device: str = "cpu", weight_decay: float = 0.0):
    device = torch.device(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            avg_val = np.mean(val_losses)
            if avg_val < best_val:
                best_val = avg_val
                best_state = model.state_dict()
            print(f"Epoch {ep:03d} | train_mse {avg_train:.6f} | val_mse {avg_val:.6f}")
        else:
            print(f"Epoch {ep:03d} | train_mse {avg_train:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_regression(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    return {"MAE": float(mae), "RMSE": float(rmse)}
