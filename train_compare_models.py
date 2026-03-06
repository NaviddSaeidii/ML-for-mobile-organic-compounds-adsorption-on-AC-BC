from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SCRIPT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = SCRIPT_DIR / "cleaned_with_deltaPZCpH_no planar.xlsx"
TARGET_COL = "log Kd (L/kg)"
TEST_SIZE = 0.2
SEED = 42

FIGSIZE = (10, 8)
DPI = 300
AX_MIN = 2.0
AX_MAX = 7.0
POINT_SIZE = 80

MODEL_ORDER = ["Random Forest", "Ridge", "SVR"]
MARKERS = {"Random Forest": "o", "Ridge": "s", "SVR": "^"}
COLORS = {"Random Forest": "#1f77b4", "Ridge": "#2ca02c", "SVR": "#d62728"}

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 13


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_unique_columns(columns) -> list[str]:
    counts = {}
    new_cols = []
    for col in columns:
        counts[col] = counts.get(col, 0) + 1
        if counts[col] == 1:
            new_cols.append(col)
        else:
            new_cols.append(f"{col}.{counts[col]}")
    return new_cols


def find_charge_columns(df: pd.DataFrame) -> tuple[str, str]:
    positive_candidates = [col for col in df.columns if "positive charge" in col.lower()]
    negative_candidates = [col for col in df.columns if "negative charge" in col.lower()]
    if not positive_candidates or not negative_candidates:
        raise RuntimeError(
            "Could not detect the positive and negative charge columns automatically. "
            "Please include columns such as 'positive charges' and 'negative charges'."
        )
    return positive_candidates[0], negative_candidates[0]


def build_feature_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the Excel file.")

    work = df.copy()

    if "number of aromatic rings" not in work.columns:
        raise RuntimeError("'number of aromatic rings' column not found.")

    positive_col, negative_col = find_charge_columns(work)

    work["number of aromatic rings"] = pd.to_numeric(work["number of aromatic rings"], errors="coerce")
    work[positive_col] = pd.to_numeric(work[positive_col], errors="coerce")
    work[negative_col] = pd.to_numeric(work[negative_col], errors="coerce")

    # Engineer manuscript-style variables
    work["has_aromatic_ring"] = (work["number of aromatic rings"] >= 1).astype(int)
    work["has_two_aromatics"] = (work["number of aromatic rings"] >= 2).astype(int)
    work["charge_state_anion"] = ((work[positive_col] == 0) & (work[negative_col] == 1)).astype(int)
    work["charge_state_cation"] = ((work[positive_col] == 1) & (work[negative_col] == 0)).astype(int)
    work["charge_state_neutral"] = ((work[positive_col] == 0) & (work[negative_col] == 0)).astype(int)
    work["charge_state_zwitterionic"] = ((work[positive_col] == 1) & (work[negative_col] == 1)).astype(int)

    # Remove raw columns converted to engineered form
    work = work.drop(columns=[positive_col, negative_col, "number of aromatic rings"], errors="ignore")

    X = work.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    X.columns = make_unique_columns(X.columns)
    return X


def load_dataset(excel_path: Path, target_col: str):
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path.resolve()}")

    df = pd.read_excel(excel_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

    X = build_feature_matrix(df, target_col)
    y = df[target_col].astype(float)

    # Keep only rows retained in X after numeric feature extraction
    merged = pd.concat([X, y.rename(target_col)], axis=1).dropna(axis=0, how="any").copy()
    X = merged.drop(columns=[target_col])
    y = merged[target_col].astype(float)

    return df, X, y


def fit_models(X_train_s: pd.DataFrame, y_train: pd.Series):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Ridge": Ridge(),
        "SVR": SVR(),
    }

    for model in models.values():
        model.fit(X_train_s, y_train)

    return models


def evaluate_models(models: dict, X_test_s: pd.DataFrame, y_test: pd.Series):
    predictions = {}
    r2_scores = {}
    rmse_scores = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        predictions[name] = y_pred
        r2_scores[name] = r2_score(y_test, y_pred)
        rmse_scores[name] = rmse(y_test, y_pred)

    return predictions, r2_scores, rmse_scores


def save_metrics(r2_scores: dict, rmse_scores: dict) -> None:
    metrics_df = pd.DataFrame({
        "Model": list(r2_scores.keys()),
        "R2": [r2_scores[name] for name in r2_scores],
        "RMSE": [rmse_scores[name] for name in rmse_scores],
    }).sort_values("Model")
    metrics_df.to_csv("true_metrics_all_models_repeat2_seed42.csv", index=False, float_format="%.6f")
    print(metrics_df.to_string(index=False))


def plot_single_model(y_test: pd.Series, predictions: dict, r2_scores: dict, rmse_scores: dict, model_name: str) -> None:
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], "k--", lw=3, label="1:1 line")

    plt.scatter(
        y_test,
        predictions[model_name],
        s=POINT_SIZE,
        marker=MARKERS[model_name],
        color=COLORS[model_name],
        edgecolor="black",
        linewidths=0.6,
        alpha=0.9,
        label=f"{model_name} (R² = {r2_scores[model_name]:.2f}, RMSE = {rmse_scores[model_name]:.2f})",
    )

    plt.xlabel("Actual log Kd (L/kg)")
    plt.ylabel("Predicted log Kd (L/kg)")
    plt.xlim(AX_MIN, AX_MAX)
    plt.ylim(AX_MIN, AX_MAX)
    plt.title(f"{model_name}: Predicted vs Actual log Kd | N = {len(y_test)}")
    plt.grid(True, linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    filename = f"predicted_vs_actual_logKd_final_repeat2_seed42_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main() -> None:
    _, X, y = load_dataset(EXCEL_PATH, TARGET_COL)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )

    models = fit_models(X_train_s, y_train)
    predictions, r2_scores, rmse_scores = evaluate_models(models, X_test_s, y_test)

    save_metrics(r2_scores, rmse_scores)

    for model_name in MODEL_ORDER:
        plot_single_model(y_test, predictions, r2_scores, rmse_scores, model_name)

    print("Saved: true_metrics_all_models_repeat2_seed42.csv")


if __name__ == "__main__":
    main()
