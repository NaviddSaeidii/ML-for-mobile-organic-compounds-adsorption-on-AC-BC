
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import shap

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)  # local RNG for reproducibility

TARGET = "log Kd (L/kg)"
EXCEL_NAME = "cleaned_with_deltaPZCpH_no planar.xlsx"

# Plot style
plt.rcParams.update({
    "figure.dpi": 300,
    "axes.grid": True
})

# ---------------------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, EXCEL_NAME)

df = pd.read_excel(file_path)
print("Actual column names:", df.columns.tolist())

if TARGET not in df.columns:
    raise RuntimeError(f"Target column '{TARGET}' not found in Excel.")

print("Target column found.")

# ---------------------------------------------------------------------
# 2) Minimal feature engineering (as before; no imputation)
# ---------------------------------------------------------------------
if "number of aromatic rings" not in df.columns:
    raise RuntimeError("'number of aromatic rings' column not found.")

df["has_aromatic_ring"] = (df["number of aromatic rings"] >= 1).astype(int)
df["has_two_aromatics"] = (df["number of aromatic rings"] >= 2).astype(int)

pos_candidates = [c for c in df.columns if "positive charge" in c.lower()]
neg_candidates = [c for c in df.columns if "negative charge" in c.lower()]
if not pos_candidates or not neg_candidates:
    raise RuntimeError("Could not auto-detect positive/negative charge columns.")
pos_col, neg_col = pos_candidates[0], neg_candidates[0]

df["charge_state_anion"]        = ((df[pos_col] == 0) & (df[neg_col] == 1)).astype(int)
df["charge_state_cation"]       = ((df[pos_col] == 1) & (df[neg_col] == 0)).astype(int)
df["charge_state_neutral"]      = ((df[pos_col] == 0) & (df[neg_col] == 0)).astype(int)
df["charge_state_zwitterionic"] = ((df[pos_col] == 1) & (df[neg_col] == 1)).astype(int)

# drop raw charge columns
df = df.drop(columns=[pos_col, neg_col])

# Base features exclude target and raw ring count; add engineered columns explicitly
base_features = df.drop(columns=[
    TARGET, 
    "number of aromatic rings",
    "has_aromatic_ring", "has_two_aromatics",
    "charge_state_anion","charge_state_cation","charge_state_neutral","charge_state_zwitterionic"
], errors="ignore")

X_rf = pd.concat([
    base_features,
    df[["has_aromatic_ring","has_two_aromatics",
        "charge_state_anion","charge_state_cation","charge_state_neutral","charge_state_zwitterionic"]]
], axis=1)

y = df[TARGET].astype(float)

# Ensure unique column names
def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

X_rf.columns = make_unique_columns(X_rf.columns)

# ---------------------------------------------------------------------
# 3) STRICT: No imputation/filling. Drop rows with any NaN or inf.
# ---------------------------------------------------------------------
data = pd.concat([X_rf, y], axis=1).replace([np.inf, -np.inf], np.nan)
before_n = len(data)
data = data.dropna(axis=0, how="any").copy()
after_n = len(data)
print(f"Dropped {before_n - after_n} rows due to NaN/inf. Kept {after_n}.")

X = data.drop(columns=[TARGET])
y = data[TARGET].astype(float)

# ---------------------------------------------------------------------
# 4) Single train/test split, reused consistently
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---------------------------------------------------------------------
# 5) Scale (fit only on TRAIN) + Random Forest (seeded)
# ---------------------------------------------------------------------
scaler = StandardScaler()
# Fit on DataFrame to keep feature names; transform DataFrames
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),  index=X_test.index,  columns=X_test.columns)

rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)

r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"\nRandom Forest (SEED={SEED}): R² = {r2:.4f}, RMSE = {rmse:.4f}")

# ---------------------------------------------------------------------
# 6) Y-randomization (use local RNG; no global np.random.seed)
# ---------------------------------------------------------------------
print("\n--- Y-randomization Test ---")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
])
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

shuffled_r2 = []
for i in range(10):
    y_shuffled = rng.permutation(y.values)
    r2_shuffled = cross_val_score(pipe, X, y_shuffled, cv=kf, scoring="r2", n_jobs=-1).mean()
    shuffled_r2.append(r2_shuffled)
print("Shuffled R² scores:", np.round(shuffled_r2, 3))
print("Mean shuffled R²:", float(np.mean(shuffled_r2)))

# ---------------------------------------------------------------------
# 7) Applicability Domain (Mahalanobis on scaled TRAIN distribution)
# ---------------------------------------------------------------------
print("\n--- Applicability Domain ---")
cov = EmpiricalCovariance().fit(X_train_s)
distances = cov.mahalanobis(X_test_s)
plt.figure()
plt.hist(distances, bins=30, edgecolor='black')
plt.title("Mahalanobis Distance of Test Set")
plt.xlabel("Distance"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig("applicability_domain_hist.png"); plt.close()

# ---------------------------------------------------------------------
# 8) Uncertainty (std dev over trees) - per-tree predict on numpy arrays to avoid feature-name warnings
# ---------------------------------------------------------------------
print("\n--- Uncertainty Estimation (std dev over trees) ---")
X_test_np = X_test_s.to_numpy()
all_preds = np.stack([tree.predict(X_test_np) for tree in rf.estimators_])
std_devs = np.std(all_preds, axis=0)
plt.figure()
plt.scatter(y_test, y_pred, c=std_devs, cmap="viridis", edgecolor='k')
plt.colorbar(label="Prediction Std Dev")
plt.xlabel("Actual log Kd"); plt.ylabel("Predicted log Kd")
plt.title("Uncertainty in Predictions (RF)")
plt.tight_layout(); plt.savefig("rf_prediction_uncertainty.png"); plt.close()

# ---------------------------------------------------------------------
# 9) Residual analysis tables/plots
# ---------------------------------------------------------------------
residuals = y_test - y_pred
outlier_threshold = 2 * np.std(residuals)
outliers = np.abs(residuals) > outlier_threshold
print(f"Found {int(outliers.sum())} outliers (|residual| > {outlier_threshold:.3f})")

meta_cols = [
    "number of aromatic rings",
    "has_aromatic_ring","has_two_aromatics",
    "charge_state_anion","charge_state_cation",
    "charge_state_neutral","charge_state_zwitterionic"
]

res_df = pd.DataFrame(index=y_test.index)
res_df["Actual"]    = y_test.values
res_df["Predicted"] = y_pred
res_df["Residual"]  = residuals
res_df["AbsError"]  = np.abs(residuals)

# attach meta from the CLEANED df (same indices exist since we dropped before splitting)
res_df = res_df.join(df[meta_cols], how="left")

def _charge_label_from_onehots(row):
    if row.get("charge_state_anion",0)==1: return "Anionic"
    if row.get("charge_state_cation",0)==1: return "Cationic"
    if row.get("charge_state_neutral",0)==1: return "Neutral"
    if row.get("charge_state_zwitterionic",0)==1: return "Zwitterionic"
    return "Unlabeled"

res_df["ChargeClass"] = res_df.apply(_charge_label_from_onehots, axis=1)
res_df["Aromaticity"] = np.where(res_df["number of aromatic rings"]>=1, "Aromatic ring", "No ring")

# Save residual table
res_df.to_csv("test_residuals_with_meta.csv")

# Boxplots with seaborn
plt.figure(figsize=(7,5))
sns.boxplot(data=res_df, x="ChargeClass", y="Residual")
plt.axhline(0, ls="--", lw=1, color="k")
plt.ylabel("Residual (Actual − Predicted)")
plt.title("Residuals by Charge Class")
plt.tight_layout(); plt.savefig("residuals_box_by_charge.png"); plt.close()

plt.figure(figsize=(6,5))
sns.boxplot(data=res_df, x="Aromaticity", y="Residual")
plt.axhline(0, ls="--", lw=1, color="k")
plt.ylabel("Residual (Actual − Predicted)")
plt.title("Residuals by Aromaticity")
plt.tight_layout(); plt.savefig("residuals_box_by_aromaticity.png"); plt.close()

# ---------------------------------------------------------------------
# 10) SHAP (Tree-based, fit explainer on TRAIN)
# ---------------------------------------------------------------------
print("\n--- SHAP ---")
try:
    explainer = shap.Explainer(rf, X_train_s)  # fit on train distribution
    shap_vals_test = explainer(X_test_s, check_additivity=False)
    # Dependence plot for a specific feature if present
    feature_name = "Log S (mol/L) at pH 7"
    if feature_name in X.columns:
        shap.plots.scatter(shap_vals_test[:, feature_name], color=shap_vals_test, show=False)
        plt.title(f"SHAP Dependence: {feature_name}")
        plt.tight_layout(); plt.savefig("shap_logS_dependence.png"); plt.close()
    # Summary plot
    shap.summary_plot(shap_vals_test, X_test_s, show=False, max_display=20)
    plt.tight_layout(); plt.savefig("shap_summary_plot.png"); plt.close()
except Exception as e:
    print("SHAP failed or is unavailable:", repr(e))

# ---------------------------------------------------------------------
# 11) Partial Dependence Plots
# ---------------------------------------------------------------------
print("\n--- Partial Dependence Plots ---")
selected_features = ["Log S (mol/L) at pH 7", "molecular weight (g/mol)", "O wt%", "surface area (m2/g)", "delta_PZCpH"]
sel_idx = [i for i, c in enumerate(X.columns) if c in selected_features]
if len(sel_idx) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(rf, X_test_s, features=sel_idx, feature_names=list(X.columns), ax=ax)
    plt.tight_layout(); plt.savefig("partial_dependence_plots_selected.png"); plt.close()
else:
    print("Selected features not all found; skipping selected PDP.")

# For all features (may be large)
fig, ax = plt.subplots(figsize=(16, 12))
PartialDependenceDisplay.from_estimator(rf, X_test_s, features=list(range(X.shape[1])), feature_names=list(X.columns), ax=ax)
plt.tight_layout(); plt.savefig("partial_dependence_plots_all_features.png"); plt.close()

# ---------------------------------------------------------------------
# 12) Distance Correlation (test set) + Matplotlib barh
# ---------------------------------------------------------------------
print("\n--- Distance Correlation with log Kd (test set) ---")
def _distance_covariance(x, y):
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    n = x.shape[0]
    a = np.abs(x.reshape(n,1) - x.reshape(1,n))
    b = np.abs(y.reshape(n,1) - y.reshape(1,n))
    A = a - a.mean(axis=0)[None,:] - a.mean(axis=1)[:,None] + a.mean()
    B = b - b.mean(axis=0)[None,:] - b.mean(axis=1)[:,None] + b.mean()
    dcov2 = (A*B).sum() / (n*n)
    return np.sqrt(max(dcov2, 0))

def _distance_correlation(x, y):
    dcov_xy = _distance_covariance(x, y)
    dcov_xx = _distance_covariance(x, x)
    dcov_yy = _distance_covariance(y, y)
    denom = np.sqrt(dcov_xx * dcov_yy)
    return float(dcov_xy / denom) if denom > 0 else 0.0

dcor_results = []
for col in X_test_s.columns:
    d = _distance_correlation(np.asarray(X_test_s[col]).ravel(), np.asarray(y_test).ravel())
    dcor_results.append((col, d))

dcor_df = pd.DataFrame(dcor_results, columns=["Feature", "Distance Correlation"]).sort_values("Distance Correlation", ascending=False)
dcor_df.to_excel("distance_correlation_results.xlsx", index=False)

plt.figure(figsize=(10, max(6, 0.3*len(dcor_df))))
plt.barh(dcor_df["Feature"], dcor_df["Distance Correlation"])
plt.gca().invert_yaxis()
plt.title("Distance Correlation with log Kd (test)")
plt.xlabel("Distance Correlation")
plt.tight_layout(); plt.savefig("distance_correlation_plot.png"); plt.close()

# ---------------------------------------------------------------------
# 13) Mutual Information (train set) + Matplotlib barh
# ---------------------------------------------------------------------
print("\n--- Mutual Information (train set) ---")
mi_scores = mutual_info_regression(X_train_s.values, y_train.values, random_state=SEED)
mi_df = pd.DataFrame({"Feature": X_train_s.columns, "Mutual Information": mi_scores}).sort_values("Mutual Information", ascending=False)
mi_df.to_excel("mutual_information_scores.xlsx", index=False)

plt.figure(figsize=(10, max(6, 0.3*len(mi_df))))
plt.barh(mi_df["Feature"], mi_df["Mutual Information"])
plt.gca().invert_yaxis()
plt.title("Mutual Information (train, scaled features)")
plt.xlabel("Mutual Information")
plt.tight_layout(); plt.savefig("mutual_information_plot.png"); plt.close()

# ---------------------------------------------------------------------
# 14) Clustering (KMeans) on processed ALL data (scaled with TRAIN scaler) - aligned indices
# ---------------------------------------------------------------------
print("\n--- Clustering with KMeans ---")
X_all_proc = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEED)
clusters = kmeans.fit_predict(X_all_proc)

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_all_proc.values)
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="Set1", edgecolor="k")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("KMeans Clusters (PCA view)")
plt.tight_layout(); plt.savefig("clustering_kmeans_pca.png"); plt.close()

# Optional: clusters on two original axes if they exist (dropna, no fills) with aligned indices
x_name = "log D (L/kg)"
y_name = "estimated log KOC  (L/kg)"
if x_name in df.columns and y_name in df.columns:
    aligned_xy = df.loc[X.index, [x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cluster_map = pd.Series(clusters, index=X.index)
    aligned_clusters = cluster_map.loc[aligned_xy.index].to_numpy()
    plt.figure(figsize=(6,5))
    plt.scatter(aligned_xy[x_name], aligned_xy[y_name], c=aligned_clusters, cmap="Set1", edgecolor="k")
    plt.xlabel(x_name); plt.ylabel(y_name)
    plt.title("KMeans Clusters on Original Axes (aligned)")
    plt.tight_layout(); plt.savefig("clustering_kmeans_original_axes.png"); plt.close()

# ---------------------------------------------------------------------
# 15) Permutation Importance (test set)
# ---------------------------------------------------------------------
print("\n--- Permutation Feature Importance (test set) ---")
perm = permutation_importance(rf, X_test_s, y_test, n_repeats=30, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame({
    "Feature": X_test_s.columns,
    "Importance Mean": perm.importances_mean,
    "Importance Std": perm.importances_std
}).sort_values("Importance Mean", ascending=True)

plt.figure(figsize=(10, max(6, 0.3*len(perm_df))))
plt.barh(perm_df["Feature"], perm_df["Importance Mean"], xerr=perm_df["Importance Std"])
plt.xlabel("Permutation Importance (mean)")
plt.title("Permutation Feature Importance (RF)")
plt.tight_layout(); plt.savefig("permutation_importance_rf.png"); plt.close()

# ---------------------------------------------------------------------
# 16) Learning Curve (pipeline without imputation)
# ---------------------------------------------------------------------
print("\n--- Learning Curve ---")
lc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
])
train_sizes, train_scores, test_scores = learning_curve(
    estimator=lc_pipe, X=X, y=y, cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True, random_state=SEED, n_jobs=-1
)
plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1),  'o-', label="CV score")
plt.xlabel("Training set size"); plt.ylabel("R² score")
plt.title("Learning Curve (RF, no impute)"); plt.legend()
plt.tight_layout(); plt.savefig("learning_curve_rf.png"); plt.close()

# ---------------------------------------------------------------------
# 17) Save model + scaler
# ---------------------------------------------------------------------
joblib.dump(rf, "rf_model_seed42_no_impute.pkl")
joblib.dump(scaler, "scaler_seed42_no_impute.pkl")
print("Saved rf_model_seed42_no_impute.pkl and scaler_seed42_no_impute.pkl")

print("\nAll done. No imputation performed. Single seed used consistently (SEED=42).")


# ================== Adsorbent-property influence by compound (SHAP-based) ==================
# Requires variables from the main run: shap_vals_test, X_test_s, y_test, y_pred.
# Computes per-compound influence from (surface area, O wt%, delta_PZCpH) using absolute SHAP.

import numpy as np, pandas as pd, matplotlib.pyplot as plt

_adsorbent_features = ["surface area (m2/g)", "O wt%", "delta_PZCpH"]
if hasattr(X_test_s, "columns") and "surface area (m2/g)" not in X_test_s.columns and "surface area (m^2/g)" in X_test_s.columns:
    _adsorbent_features[0] = "surface area (m^2/g)"

_missing_ads = [f for f in _adsorbent_features if (not hasattr(X_test_s, "columns")) or (f not in X_test_s.columns)]
if _missing_ads:
    print("WARNING: some adsorbent features are missing from model inputs:", _missing_ads)

# Convert SHAP output to matrix (n_samples x n_features)
_shap_mat = shap_vals_test.values if hasattr(shap_vals_test, "values") else np.array(shap_vals_test)
_abs_shap = np.abs(_shap_mat)

# Per-feature absolute SHAP arrays for the adsorbent properties
_per_feat_abs = {}
if hasattr(X_test_s, "columns"):
    for _f in _adsorbent_features:
        if _f in X_test_s.columns:
            _per_feat_abs[_f] = _abs_shap[:, list(X_test_s.columns).index(_f)]

if len(_per_feat_abs) > 0:
    _ads_influence = np.sum([_per_feat_abs[_f] for _f in _per_feat_abs], axis=0)
    _total_abs = _abs_shap.sum(axis=1)
    _fraction  = np.divide(_ads_influence, _total_abs, out=np.zeros_like(_ads_influence), where=_total_abs>0)

    _influence_df = pd.DataFrame(index=y_test.index)
    _influence_df["Actual"]    = y_test.values
    _influence_df["Predicted"] = y_pred
    _influence_df["Residual"]  = y_test.values - y_pred
    for _f in _per_feat_abs:
        _influence_df[f"ABS_SHAP__{_f}"] = _per_feat_abs[_f]
    _influence_df["ABS_SHAP__AdsorbentSum"] = _ads_influence
    _influence_df["ABS_SHAP__Total"]        = _total_abs
    _influence_df["AdsorbentInfluence_Fraction"] = _fraction

    _sorted = _influence_df.sort_values("ABS_SHAP__AdsorbentSum", ascending=False)
    _sorted.to_csv("adsorbent_influence_by_compound.csv", index=True)
    print("Saved: adsorbent_influence_by_compound.csv")

    # Plot 1: Top compounds by absolute adsorbent influence
    _topN = min(15, len(_sorted))
    plt.figure(figsize=(8, 6))
    plt.barh(_sorted.index.astype(str)[:_topN], _sorted["ABS_SHAP__AdsorbentSum"][:_topN])
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute SHAP (sum of surface area, O wt%, ΔPZCpH)")
    plt.title("Top compounds most influenced by adsorbent properties")
    plt.tight_layout(); plt.savefig("top_compounds_adsorbent_influence.png", dpi=300); plt.close()

    # Plot 2: Actual vs Predicted colored by fraction of adsorbent influence
    plt.figure(figsize=(7, 6))
    _sc = plt.scatter(_influence_df["Actual"], _influence_df["Predicted"],
                      c=_influence_df["AdsorbentInfluence_Fraction"],
                      cmap="viridis", edgecolor="k")
    plt.colorbar(_sc, label="Fraction of total |SHAP| from adsorbent features")
    _mn, _mx = float(_influence_df["Actual"].min()), float(_influence_df["Actual"].max())
    plt.plot([_mn, _mx], [_mn, _mx], "k--", lw=1)
    plt.xlabel("Actual log Kd (L/kg)")
    plt.ylabel("Predicted log Kd (L/kg)")
    plt.title("Per-compound influence of adsorbent properties")
    plt.tight_layout(); plt.savefig("actual_vs_pred_colored_by_adsorbent_influence.png", dpi=300); plt.close()
    print("Saved figures: top_compounds_adsorbent_influence.png, actual_vs_pred_colored_by_adsorbent_influence.png")
else:
    print("Adsorbent features not present in model inputs; skipping influence analysis.")
# ================== End adsorbent-property influence section ==================




# ================== NEW: Aggregate adsorbent influence by chemistry categories ==================
# This section groups the per-compound adsorbent influence by:
#   - charge class: anionic / cationic / neutral / zwitterionic
#   - aromaticity: aromatic / nonaromatic
#   - cross-category (8 groups): e.g., "anionic aromatic", "neutral nonaromatic", etc.
# It requires that the adsorbent influence table _influence_df has been built above.

def _label_charge(row):
    if row.get("charge_state_anion",0)==1: return "anionic"
    if row.get("charge_state_cation",0)==1: return "cationic"
    if row.get("charge_state_neutral",0)==1: return "neutral"
    if row.get("charge_state_zwitterionic",0)==1: return "zwitterionic"
    return "unlabeled"

# Attach labels to the source df (aligned indices to X)
df["charge_class"] = df.apply(_label_charge, axis=1)
df["aromaticity"]  = np.where(df["number of aromatic rings"]>=1, "aromatic", "nonaromatic")

if '_influence_df' in locals():
    _influence_df = _influence_df.join(df[["charge_class","aromaticity"]], how="left")
    _influence_df["charge_x_aromatic"] = _influence_df["charge_class"] + " " + _influence_df["aromaticity"]

    # Save enriched table
    _influence_df.to_csv("adsorbent_influence_by_compound_WITH_CATEGORIES.csv")

    # 1) Mean absolute SHAP by charge class
    _mean_by_charge = (_influence_df.groupby("charge_class")["ABS_SHAP__AdsorbentSum"]
                       .mean().sort_values(ascending=False))
    plt.figure(figsize=(7,5))
    plt.bar(_mean_by_charge.index, _mean_by_charge.values)
    plt.ylabel("Mean absolute SHAP (adsorbent features)")
    plt.title("Adsorbent influence by charge class")
    plt.tight_layout(); plt.savefig("shap_adsorbent_by_charge_mean.png"); plt.close()

    # 2) Mean absolute SHAP by aromaticity
    _mean_by_arom = (_influence_df.groupby("aromaticity")["ABS_SHAP__AdsorbentSum"]
                     .mean().sort_values(ascending=False))
    plt.figure(figsize=(6,5))
    plt.bar(_mean_by_arom.index, _mean_by_arom.values)
    plt.ylabel("Mean absolute SHAP (adsorbent features)")
    plt.title("Adsorbent influence by aromaticity")
    plt.tight_layout(); plt.savefig("shap_adsorbent_by_aromaticity_mean.png"); plt.close()

    # 3) Cross-category (8 groups)
    _mean_by_cross = (_influence_df.groupby("charge_x_aromatic")["ABS_SHAP__AdsorbentSum"]
                      .mean().sort_values(ascending=False))
    plt.figure(figsize=(10,5))
    plt.bar(_mean_by_cross.index, _mean_by_cross.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean absolute SHAP (adsorbent features)")
    plt.title("Adsorbent influence by charge × aromaticity")
    plt.tight_layout(); plt.savefig("shap_adsorbent_by_crosscategory_mean.png"); plt.close()

    # 4) Stacked contributions per cross-category
    _per_feat_cols = [c for c in _influence_df.columns if c.startswith("ABS_SHAP__") and "AdsorbentSum" not in c and "Total" not in c]
    _comp_means = _influence_df.groupby("charge_x_aromatic")[_per_feat_cols].mean().loc[_mean_by_cross.index]
    plt.figure(figsize=(10,5))
    _bottom = np.zeros(len(_comp_means))
    for _col in _comp_means.columns:
        plt.bar(_comp_means.index, _comp_means[_col].values, bottom=_bottom, label=_col.replace("ABS_SHAP__",""))
        _bottom += _comp_means[_col].values
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean absolute SHAP")
    plt.title("Decomposition of adsorbent influence by cross category")
    plt.legend(title="Feature")
    plt.tight_layout(); plt.savefig("shap_adsorbent_crosscategory_stacked.png"); plt.close()

    # 5) Top compounds figure with category labels
    _topN = min(15, len(_influence_df))
    _sorted = _influence_df.sort_values("ABS_SHAP__AdsorbentSum", ascending=False).head(_topN).copy()
    _labels = [f"{idx} ({row['charge_x_aromatic']})" for idx, row in _sorted.iterrows()]
    plt.figure(figsize=(9,6))
    plt.barh(_labels, _sorted["ABS_SHAP__AdsorbentSum"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute SHAP (sum of surface area, O wt%, ΔPZCpH)")
    plt.title("Top compounds most influenced by adsorbent properties (with categories)")
    plt.tight_layout(); plt.savefig("top_compounds_adsorbent_influence_WITH_CATEGORIES.png"); plt.close()

# ================== END NEW SECTION ==================
