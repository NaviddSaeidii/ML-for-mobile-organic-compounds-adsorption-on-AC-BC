from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

# Default files for the GitHub version
TRAINING_CSV = SCRIPT_DIR / "merged_train_external.csv"
INPUT_EXCEL = SCRIPT_DIR / "Prediction_Input_Template.xlsx"

TARGET_COL = "log Kd (L/kg)"
PRED_COL = "pred_log Kd (L/kg)"

MODEL_PATH = SCRIPT_DIR / "rf_model_seed42_no_impute.joblib"
SCALER_PATH = SCRIPT_DIR / "scaler_seed42_no_impute.joblib"
FEATS_PATH = SCRIPT_DIR / "feature_names_seed42_no_impute.joblib"

SEED = 42

MOLECULAR_FEATURE_ORDER = [
    "log D (L/kg)",
    "molecular weight (g/mol)",
    "Log S (mol/L) at pH 7",
    "molecular volume  (cm3/mol/100)",
    "estimated log KOC  (L/kg)",
    "has_aromatic_ring",
    "has_two_aromatics",
    "charge_state_anion",
    "charge_state_cation",
    "charge_state_neutral",
    "charge_state_zwitterionic",
]

ADSORBENT_FEATURE_ORDER = [
    "surface area (m2/g)",
    "O wt%",
    "delta_PZCpH",
]

FEATURE_ORDER = MOLECULAR_FEATURE_ORDER + ADSORBENT_FEATURE_ORDER


def _read_training_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV file not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # If a split column exists in another dataset version, keep only training rows
    if "data_split" in df.columns:
        df = df.loc[df["data_split"] == "train_literature"].copy()

    return df


def _read_excel_frame(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path.resolve()}")

    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_charge_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols_lower = {col.lower(): col for col in df.columns}

    positive_candidates = [
        col for col in df.columns
        if "positive charge" in col.lower() or "positive charges" in col.lower()
    ]
    negative_candidates = [
        col for col in df.columns
        if "negative charge" in col.lower() or "negative charges" in col.lower()
    ]

    if not positive_candidates or not negative_candidates:
        raise ValueError(
            "Could not detect the positive and negative charge columns automatically. "
            "Please make sure the file contains columns such as "
            "'positive charges' and 'negative charges'."
        )

    return positive_candidates[0], negative_candidates[0]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    positive_col, negative_col = _find_charge_columns(work)

    required_raw_cols = [
        "number of aromatic rings",
        "log D (L/kg)",
        "molecular weight (g/mol)",
        "Log S (mol/L) at pH 7",
        "molecular volume  (cm3/mol/100)",
        "estimated log KOC  (L/kg)",
        "surface area (m2/g)",
        "O wt%",
        "delta_PZCpH",
    ]

    missing_raw = [c for c in required_raw_cols if c not in work.columns]
    if missing_raw:
        raise ValueError(f"Missing required input columns: {missing_raw}")

    work["number of aromatic rings"] = pd.to_numeric(
        work["number of aromatic rings"], errors="coerce"
    )
    work[positive_col] = pd.to_numeric(work[positive_col], errors="coerce")
    work[negative_col] = pd.to_numeric(work[negative_col], errors="coerce")

    work["has_aromatic_ring"] = (work["number of aromatic rings"] >= 1).astype(float)
    work["has_two_aromatics"] = (work["number of aromatic rings"] >= 2).astype(float)

    work["charge_state_anion"] = (
        (work[positive_col] == 0) & (work[negative_col] >= 1)
    ).astype(float)

    work["charge_state_cation"] = (
        (work[positive_col] >= 1) & (work[negative_col] == 0)
    ).astype(float)

    work["charge_state_neutral"] = (
        (work[positive_col] == 0) & (work[negative_col] == 0)
    ).astype(float)

    work["charge_state_zwitterionic"] = (
        (work[positive_col] >= 1) & (work[negative_col] >= 1)
    ).astype(float)

    return work


def _make_training_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Training CSV must contain the target column '{TARGET_COL}'.")

    work = df.replace([np.inf, -np.inf], np.nan).copy()
    work = _engineer_features(work)

    X = work[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work[TARGET_COL], errors="coerce")

    data = pd.concat([X, y.rename(TARGET_COL)], axis=1).dropna(axis=0, how="any").copy()

    if data.empty:
        raise ValueError("No valid training rows remain after removing rows with missing values.")

    return data.drop(columns=[TARGET_COL]), data[TARGET_COL].astype(float)


def train_and_save(csv_path: Path):
    df = _read_training_csv(csv_path)
    X, y = _make_training_Xy(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATS_PATH)

    return rf, scaler, list(X.columns)


def ensure_model(training_path: Path, force_retrain: bool = False):
    have_all = MODEL_PATH.exists() and SCALER_PATH.exists() and FEATS_PATH.exists()

    if force_retrain or not have_all:
        return train_and_save(training_path)

    rf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats = joblib.load(FEATS_PATH)
    return rf, scaler, feats


def predict_excel(input_excel: Path, training_csv: Path, force_retrain: bool = False):
    rf, scaler, feats = ensure_model(training_csv, force_retrain=force_retrain)

    raw = _read_excel_frame(input_excel)
    work = _engineer_features(raw)

    missing = [c for c in feats if c not in work.columns]
    if missing:
        raise ValueError(f"Input Excel is missing required columns: {missing}")

    X = work[feats].apply(pd.to_numeric, errors="coerce")
    bad_rows = X.isna().any(axis=1)

    if bad_rows.any():
        bad_row_indices = list(raw.index[bad_rows])
        raise ValueError(
            f"Rows with missing or non-numeric required inputs: {bad_row_indices}"
        )

    preds = rf.predict(scaler.transform(X))

    backup = input_excel.with_name(input_excel.stem + ".backup.xlsx")
    shutil.copy2(input_excel, backup)

    out = raw.copy()
    out[PRED_COL] = preds
    out.to_excel(input_excel, index=False)

    return input_excel, backup, len(out)


def main():
    parser = argparse.ArgumentParser(
        description="Predict log Kd values from an Excel input file using a Random Forest model."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining from the CSV dataset before prediction.",
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_EXCEL),
        help="Excel file with prediction inputs.",
    )
    parser.add_argument(
        "--training",
        default=str(TRAINING_CSV),
        help="CSV file with training data.",
    )
    args = parser.parse_args()

    input_excel = Path(args.input)
    if not input_excel.is_absolute():
        input_excel = SCRIPT_DIR / input_excel

    training_csv = Path(args.training)
    if not training_csv.is_absolute():
        training_csv = SCRIPT_DIR / training_csv

    updated, backup, nrows = predict_excel(
        input_excel=input_excel,
        training_csv=training_csv,
        force_retrain=args.train,
    )

    print(f"Predictions written to: {updated}")
    print(f"Backup created: {backup}")
    print(f"Rows predicted: {nrows}")


if __name__ == "__main__":
    main()