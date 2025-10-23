# trainers/arima_trainer.py
from typing import Tuple, Dict
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_arima(train_series, test_series, order=(5,1,0)):
    model = ARIMA(train_series, order=order)
    res = model.fit()
    n_forecast = len(test_series)
    pred = res.forecast(steps=n_forecast)
    mae = mean_absolute_error(test_series.values, pred.values)
    rmse = mean_squared_error(test_series.values, pred.values, squared=False)
    metrics = {"MAE": float(mae), "RMSE": float(rmse)}
    return res, pred, metrics
