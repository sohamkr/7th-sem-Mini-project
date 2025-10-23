# Time-series Water Quality Forecasting — README

A modular, production-friendly project to build and compare time-series models (LSTM, GRU, XGBoost, ARIMA) for forecasting water-quality targets (e.g., E. coli CFU/mL) from sensor streams. The repo provides data loaders, feature engineering, training pipelines, evaluation, and model export hooks — split into small files so you can run, extend, and deploy pieces independently.

---

# Table of contents

* Project summary
* Problem statement & goals
* Key features
* Data & expected CSV schema
* Models implemented
* Pipeline overview
* Installation
* Quick start / running the pipeline
* Configuration
* How to add your data
* Evaluation & metrics
* Recommended experiments
* Deployment notes (server vs edge)
* Project structure
* Future work / extensions
* License & acknowledgements
* Contact

---

# Project summary

This project trains and evaluates multiple forecasting approaches on time-series sensor + lab data to produce short-term forecasts (e.g., 3–7 day horizon) and alerting for contamination events. The code is modular so you can iterate quickly: swap models, change features, run experiments, and export models for edge or server inference.

---

# Problem statement & goals

* Forecast a water-quality target (example: `target` column — E. coli CFU/mL) using time-series sensor streams (temperature, turbidity, pH, flow, etc.).
* Provide both accurate numeric forecasts and binary/multi-class alerts (Normal / Elevated / Outbreak).
* Compare classic statistical methods (ARIMA), tree-based tabular models (XGBoost), and sequence models (LSTM/GRU).
* Provide a practical path for deployment: simple on-device rules for microcontrollers (STM32/ESP32) and heavier models on a server.

---

# Key features

* Clean, modular codebase split into data loaders, feature engineering, PyTorch dataset and models, and trainer modules.
* Implements:

  * LSTM and GRU sequence models (PyTorch)
  * XGBoost tabular model using lag features
  * ARIMA statistical baseline
* Time-based train/test split and scaler persistence
* Outputs predictions & evaluation metrics (MAE, RMSE), and saves model artifacts
* Easy to extend for multi-step forecasting, quantile/uncertainty prediction, and anomaly detection

---

# Data & expected CSV schema

The pipeline expects a CSV with at least:

* `timestamp` — parseable datetime column (ISO or common formats)
* `target` — numeric target to forecast (e.g., E. coli counts)
* other numeric columns — sensor streams (e.g., `temp`, `turbidity`, `ph`, `flow`)

Example (CSV):

```
timestamp,target,temp,turbidity,ph,flow
2024-01-01 00:00,12.5,25.1,3.4,7.2,0.12
2024-01-02 00:00,10.4,24.8,3.5,7.1,0.11
...
```

If your column names differ, change `DATA_TIMESTAMP_COL` and `DATA_TARGET_COL` in `config.py`.

---

# Models implemented

* **LSTM (seq model)** — time-series regression using lookback windows (PyTorch).
* **GRU (seq model)** — lighter RNN with similar interface to LSTM.
* **XGBoost (tabular)** — uses target lag features (`target_lag1...target_lagN`) + current sensors.
* **ARIMA** — classical statistical baseline trained on the target series.

---

# Pipeline overview

1. Load CSV and parse timestamps.
2. Time-based train/test split (default: last 30 days as test).
3. Feature engineering:

   * For seq models: scale features + target, create sliding windows.
   * For XGBoost: create lag features and optional rolling stats.
4. Train chosen model(s).
5. Evaluate on test set with MAE & RMSE; save predictions & model artifacts.
6. (Optional) Export model for inference or further conversion to ONNX/TFLite.

---

# Installation

1. Clone or copy the project files into a folder (e.g., `time_series_models/`).
2. Install dependencies (recommended in virtualenv):

```bash
pip install -r requirements.txt
```

---

# Quick start / running the pipeline

Run the full pipeline (train LSTM, GRU, XGBoost, ARIMA and save outputs):

```bash
python run_all.py path/to/your_data.csv
```

Or, via CLI wrapper:

```bash
python cli.py --data path/to/your_data.csv --mode all
```

Outputs are written to `outputs/` (configurable in `config.py`).


# How to add your data

1. Place CSV somewhere accessible (e.g., `data/my_run.csv`).
2. Confirm timestamp format parses with `pd.to_datetime`.
3. Update `config.py` if column names differ.
4. Run the pipeline and inspect `outputs/preds_*.csv` for predictions.

---

# Evaluation & metrics

* Regression metrics:

  * **MAE** (mean absolute error)
  * **RMSE** (root mean squared error)
* For alerting/classification, later convert forecasts to classes and measure:

  * **Precision, Recall, F1** (focus on Recall for early-warning)
* Use time-series cross-validation (walk-forward) to get robust estimates.




