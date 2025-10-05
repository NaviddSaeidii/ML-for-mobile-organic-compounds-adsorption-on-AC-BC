
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import floor, ceil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# -------------------- Settings --------------------
EXCEL_PATH   = "cleaned_with_deltaPZCpH_no planar.xlsx"
TARGET_COL   = "log Kd (L/kg)"
TEST_SIZE    = 0.2
SEED         = 42

# Manuscript-like styling
mpl.rcParams["font.family"] = "Arial"   # falls back if Arial not installed
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 13

FIGSIZE = (10, 8)
DPI = 300
AX_MIN = 2.0
AX_MAX = 7.5
POINT_SIZE = 80

MARKERS = {"Random Forest": "o", "Ridge": "s", "SVR": "^"}
COLORS  = {"Random Forest": "#1f77b4", "Ridge": "#2ca02c", "SVR": "#d62728"}

# -------------------- Load & clean --------------------
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel file not found: {os.path.abspath(EXCEL_PATH)}")

df = pd.read_excel(EXCEL_PATH).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

# Split target/features (numeric-only features to avoid text columns)
y = df[TARGET_COL].astype(float)
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).copy()

# -------------------- Split once (shared by all models) --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# -------------------- Scale (fit on TRAIN only) --------------------
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),  index=X_test.index,  columns=X_test.columns)

# -------------------- Models --------------------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Ridge": Ridge(),
    "SVR": SVR(),
}

def rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -------------------- Fit & evaluate --------------------
preds, r2s, rmses = {}, {}, {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    preds[name] = y_pred
    r2s[name] = r2_score(y_test, y_pred)
    rmses[name] = rmse(y_test, y_pred)

# Save numeric metrics
metrics_df = pd.DataFrame({"R2": r2s, "RMSE": rmses}).T.sort_index()
metrics_df.to_csv("true_metrics_all_models_repeat2_seed42.csv", float_format="%.6f")
print(metrics_df)

# -------------------- Axis bounds (shared) --------------------
min_ref = min(y_test.min(), *(p.min() for p in preds.values()))
max_ref = max(y_test.max(), *(p.max() for p in preds.values()))
span = max_ref - min_ref
pad = 0.05 * span if span > 0 else 0.5
ax_min = floor((min_ref - pad) * 2) / 2.0
ax_max = ceil((max_ref + pad) * 2) / 2.0

# -------------------- Plot A: All models --------------------
markers = {
    "Random Forest": 'o',
    "Ridge": 's',
    "SVR": '^'
}

plt.figure(
figsize=FIGSIZE, dpi=DPI)
plt.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=2, label="1:1 Line")

for name in ["Random Forest", "Ridge", "SVR"]:
    plt.scatter(
        y_test, preds[name],
        s=POINT_SIZE, marker=markers[name], color=COLORS[name], edgecolor='k', linewidths=0.6, alpha=0.9,
        label=f"{name} (R² = {r2s[name]:.2f}, RMSE = {rmses[name]:.2f})"
    )

plt.xlabel("Actual log Kd (L/kg)")
plt.ylabel("Predicted log Kd (L/kg)")
plt.xlim(AX_MIN, AX_MAX)
plt.ylim(AX_MIN, AX_MAX)
plt.title(f"Predicted vs Actual log Kd — All Models | N = {len(y_test)}")
plt.xlim(ax_min, ax_max); plt.ylim(ax_min, ax_max)
plt.grid(True, linewidth=0.6, alpha=0.6)
plt.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.savefig("predicted_vs_actual_logKd_final_repeat2_seed42_all.png", dpi=DPI, bbox_inches="tight")

# -------------------- Plot B: Random Forest only --------------------
markers = {
    "Random Forest": 'o',
    "Ridge": 's',
    "SVR": '^'
}

plt.figure(
figsize=FIGSIZE, dpi=DPI)
plt.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=2, label="1:1 Line")
plt.scatter(
    y_test, preds["Random Forest"],
    s=POINT_SIZE,
    marker=MARKERS["Random Forest"],
    color=COLORS["Random Forest"],
    edgecolor="black", linewidths=0.6, alpha=0.9,
    label=f"Random Forest (R² = {r2s['Random Forest']:.2f}, RMSE = {rmses['Random Forest']:.2f})"
)
plt.xlabel("Actual log Kd (L/kg)")
plt.ylabel("Predicted log Kd (L/kg)")
plt.title(f"Random Forest: Predicted vs Actual log Kd | N = {len(y_test)}")
plt.xlim(ax_min, ax_max); plt.ylim(ax_min, ax_max)
plt.grid(True, linewidth=0.6, alpha=0.6)
plt.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.savefig("predicted_vs_actual_logKd_final_repeat2_seed42_rf.png", dpi=DPI, bbox_inches="tight")

print("Saved: predicted_vs_actual_logKd_final_repeat2_seed42_all.png")
print("Saved: predicted_vs_actual_logKd_final_repeat2_seed42_rf.png")
print("Saved: true_metrics_all_models_repeat2_seed42.csv")
