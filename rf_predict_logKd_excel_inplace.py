
#!/usr/bin/env python3
"""
Random Forest predictor for log Kd (L/kg), seed=42, NO imputation.

Usage (default filenames):
    python rf_predict_logKd_excel_inplace_no_impute_seed42.py [--train]
    # --train forces retraining from the training Excel

Artifacts created on training:
    - rf_model_seed42_no_impute.joblib
    - scaler_seed42_no_impute.joblib
    - feature_names_seed42_no_impute.joblib

Behavior:
    * Training: loads 'cleaned_with_deltaPZCpH_no planar.xlsx', keeps numeric features,
      drops rows with NaN/±inf ONLY (no filling), fits StandardScaler on TRAIN DATA,
      trains RF (n_estimators=200, random_state=42), saves artifacts.
    * Prediction: reads 'Prediction_Input_Template.xlsx', validates columns & numeric values,
      applies saved scaler/model, writes predictions to a new column
      'pred_log Kd (L/kg)' in the SAME file after making a .backup.xlsx copy.
"""

import argparse
from pathlib import Path
import warnings
import shutil

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from openpyxl import load_workbook

# ===== Filenames =====
TRAINING_EXCEL = Path("cleaned_with_deltaPZCpH_no planar.xlsx")
INPUT_EXCEL    = Path("Prediction_Input_Template.xlsx")
TARGET_COL     = "log Kd (L/kg)"
PRED_COL       = "pred_log Kd (L/kg)"

MODEL_PATH   = Path("rf_model_seed42_no_impute.joblib")
SCALER_PATH  = Path("scaler_seed42_no_impute.joblib")
FEATS_PATH   = Path("feature_names_seed42_no_impute.joblib")

SEED = 42

# ===== Utilities =====
def _read_training_frame(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Training Excel not found: {excel_path.resolve()}")
    df = pd.read_excel(excel_path)
    # Replace inf with NaN and drop any rows with missing in *any* column used
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training Excel.")
    return df

def _make_training_Xy(df: pd.DataFrame):
    # Numeric-only features; drop target from features
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    data = pd.concat([X, y], axis=1).dropna(axis=0, how="any").copy()  # NO IMPUTATION
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL].astype(float)
    if X.empty:
        raise ValueError("No numeric features found for training after cleaning.")
    return X, y

def train_and_save(excel_path: Path):
    print("Training model from:", excel_path)
    df = _read_training_frame(excel_path)
    X, y = _make_training_Xy(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)

    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATS_PATH)
    print("Saved:", MODEL_PATH.name, SCALER_PATH.name, FEATS_PATH.name)
    return rf, scaler, list(X.columns)

def load_artifacts():
    rf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats = joblib.load(FEATS_PATH)
    return rf, scaler, feats

def ensure_model(force_retrain=False):
    have_all = MODEL_PATH.exists() and SCALER_PATH.exists() and FEATS_PATH.exists()
    if force_retrain or not have_all:
        print("Training model...")
        return train_and_save(TRAINING_EXCEL)
    print("Loading existing model...")
    return load_artifacts()

# ===== Prediction helpers =====
def align_features(df_in: pd.DataFrame, required_cols):
    """Return df with required_cols in order, all numeric; raise informative errors otherwise."""
    # Ignore extra columns but warn once
    extra = [c for c in df_in.columns if c not in required_cols]
    if extra:
        warnings.warn(f"Ignoring extra columns not used by model: {extra}")

    missing = [c for c in required_cols if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required columns, coerce to numeric
    X = df_in[required_cols].copy()
    for c in required_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Identify problem cells
    bad = X.isna()
    if bad.any().any():
        rows, cols = np.where(bad.values)
        details = []
        for r, cidx in zip(rows, cols):
            details.append(f"(row {r+2}, column '{required_cols[cidx]}')")
        raise ValueError(
            "Found non-numeric or missing values in the prediction input at: "
            + ", ".join(details) +
            ".\nNo imputation is performed—please correct the input cells."
        )
    return X

def predict(rf, scaler, X_df: pd.DataFrame):
    return rf.predict(scaler.transform(X_df))

def write_predictions_inplace(xlsx_path: Path, preds):
    wb = load_workbook(xlsx_path)
    ws = wb.worksheets[0]  # first sheet

    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    if PRED_COL in headers:
        col_idx = headers.index(PRED_COL) + 1
    else:
        col_idx = len(headers) + 1
        ws.cell(row=1, column=col_idx, value=PRED_COL)

    for i, p in enumerate(preds, start=2):
        ws.cell(row=i, column=col_idx, value=float(p))

    wb.save(xlsx_path)

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Force retraining from the training Excel")
    parser.add_argument("--training", type=str, default=str(TRAINING_EXCEL), help="Path to training Excel")
    parser.add_argument("--input", type=str, default=str(INPUT_EXCEL), help="Path to prediction input Excel")
    args = parser.parse_args()

    training_path = Path(args.training)
    input_path    = Path(args.input)

    rf, scaler, feature_names = ensure_model(force_retrain=args.train)

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input Excel not found: {input_path.resolve()}")

    df_in = pd.read_excel(input_path).replace([np.inf, -np.inf], np.nan)
    X = align_features(df_in, feature_names)

    preds = predict(rf, scaler, X)

    backup = input_path.with_suffix(".backup.xlsx")
    shutil.copy2(input_path, backup)
    print(f"Backup created: {backup.name}")

    write_predictions_inplace(input_path, preds)
    print(f"Predictions written into {input_path.name}, column '{PRED_COL}'")

if __name__ == "__main__":
    main()
