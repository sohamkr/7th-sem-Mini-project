# run_all.py
import os
import numpy as np
import pandas as pd
from config import DATA_TIMESTAMP_COL, DATA_TARGET_COL, OUT_DIR, SEQ_LEN, HORIZON, BATCH_SIZE, TEST_DAYS, DEVICE
from data.loaders import load_csv, time_train_test_split
from data.features import create_lag_features, fit_scaler
from datasets import TimeSeriesDataset
from models.seq_model import SeqModel
from trainers.seq_trainer import train_seq_model, evaluate_regression
from trainers.xgb_trainer import train_xgboost
from trainers.arima_trainer import train_arima
from torch.utils.data import DataLoader
import joblib

def run_all(csv_path: str):
    df = load_csv(csv_path, timestamp_col=DATA_TIMESTAMP_COL)
    train_df, test_df = time_train_test_split(df, timestamp_col=DATA_TIMESTAMP_COL, test_size_days=TEST_DAYS)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if DATA_TARGET_COL not in numeric_cols:
        raise ValueError(f"{DATA_TARGET_COL} must be numeric and present in CSV")
    feat_cols = [c for c in numeric_cols if c != DATA_TARGET_COL]

   
    scaler = fit_scaler(train_df[feat_cols + [DATA_TARGET_COL]].values, save_path=os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    scaled_full = scaler.transform(df[feat_cols + [DATA_TARGET_COL]].values)
    scaled_df = pd.DataFrame(scaled_full, columns=feat_cols + [DATA_TARGET_COL])
    scaled_df[DATA_TIMESTAMP_COL] = df[DATA_TIMESTAMP_COL].values

    
    arr = scaled_df[feat_cols + [DATA_TARGET_COL]].values
    train_mask = scaled_df[DATA_TIMESTAMP_COL] <= train_df[DATA_TIMESTAMP_COL].max()
    arr_train = arr[train_mask.values]
    arr_test = arr[~train_mask.values]

    train_ds = TimeSeriesDataset(arr_train, seq_len=SEQ_LEN, horizon=HORIZON)
    
    test_ds = TimeSeriesDataset(np.vstack([arr_train[-SEQ_LEN:], arr_test]), seq_len=SEQ_LEN, horizon=HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {}

    for rnn in ["lstm", "gru"]:
        print(f"\nTraining {rnn}...")
        model = SeqModel(inp_dim=len(feat_cols), hidden_dim=64, n_layers=1, rnn_type=rnn, dropout=0.2)
        model = train_seq_model(model, train_loader, val_loader=test_loader, epochs=40, lr=1e-3, device=DEVICE)
      
        model.eval()
        preds, trues = [], []
        import torch
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy().flatten()
                preds.append(out)
                trues.append(yb.numpy().flatten())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        target_mean = scaler.mean_[-1]
        target_scale = np.sqrt(scaler.var_[-1])
        preds_un = preds * target_scale + target_mean
        trues_un = trues * target_scale + target_mean
        metrics = evaluate_regression(trues_un, preds_un)
        results[rnn] = {"preds": preds_un, "trues": trues_un, "metrics": metrics}
        print(f"{rnn} metrics: {metrics}")

   
    print("\nPreparing XGBoost features...")
    lags = list(range(1, SEQ_LEN+1))
    xgb_df = create_lag_features(df[[DATA_TIMESTAMP_COL] + feat_cols + [DATA_TARGET_COL]].copy(), cols=[DATA_TARGET_COL], lags=lags)
    train_xgb = xgb_df[xgb_df[DATA_TIMESTAMP_COL] <= train_df[DATA_TIMESTAMP_COL].max()].copy().reset_index(drop=True)
    test_xgb = xgb_df[xgb_df[DATA_TIMESTAMP_COL] > train_df[DATA_TIMESTAMP_COL].max()].copy().reset_index(drop=True)
    feature_columns = [f"{DATA_TARGET_COL}_lag{lag}" for lag in lags] + feat_cols
    train_xgb = train_xgb.dropna().reset_index(drop=True)
    test_xgb = test_xgb.dropna().reset_index(drop=True)
    print(f"XGBoost train rows: {len(train_xgb)} | test rows: {len(test_xgb)}")
    xgb_model, xgb_preds, xgb_metrics = train_xgboost(train_xgb, test_xgb, features=feature_columns, target_col=DATA_TARGET_COL, num_round=300)
    results['xgboost'] = {"preds": xgb_preds, "metrics": xgb_metrics}
    xgb_model.save_model(os.path.join(OUT_DIR, "xgb_model.json"))
    print(f"XGBoost metrics: {xgb_metrics}")

    
    print("\nFitting ARIMA...")
    train_series = train_df[DATA_TARGET_COL].astype(float).reset_index(drop=True)
    test_series = test_df[DATA_TARGET_COL].astype(float).reset_index(drop=True)
    try:
        arima_res, arima_pred, arima_metrics = train_arima(train_series, test_series, order=(5,1,0))
        results['arima'] = {"preds": arima_pred.values, "metrics": arima_metrics}
        print(f"ARIMA metrics: {arima_metrics}")
    except Exception as e:
        results['arima'] = {"error": str(e)}
        print("ARIMA failed:", e)

   
    print("\nSaving results...")
    for k, v in results.items():
        if 'preds' in v and 'trues' in v:
            out_df = pd.DataFrame({"true": v['trues'], "pred": v['preds']})
            out_df.to_csv(os.path.join(OUT_DIR, f"preds_{k}.csv"), index=False)
        elif 'preds' in v:
            pd.DataFrame({"pred": v['preds']}).to_csv(os.path.join(OUT_DIR, f"preds_{k}.csv"), index=False)
    print("Done. Outputs in", OUT_DIR)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_all.py path/to/data.csv")
    else:
        run_all(sys.argv[1])
