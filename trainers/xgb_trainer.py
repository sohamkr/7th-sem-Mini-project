# trainers/xgb_trainer.py
from typing import Tuple, List, Dict
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_xgboost(train_df, test_df, features: List[str], target_col: str = "target", params=None, num_round: int = 200):
    params = params or {"objective":"reg:squarederror", "verbosity":0}
    dtrain = xgb.DMatrix(train_df[features], label=train_df[target_col])
    dtest = xgb.DMatrix(test_df[features], label=test_df[target_col])
    model = xgb.train(params, dtrain, num_boost_round=num_round, evals=[(dtrain, 'train'), (dtest, 'eval')], verbose_eval=False)
    preds = model.predict(dtest)
    mae = mean_absolute_error(test_df[target_col].values, preds)
    rmse = mean_squared_error(test_df[target_col].values, preds, squared=False)
    metrics = {"MAE": float(mae), "RMSE": float(rmse)}
    return model, preds, metrics
