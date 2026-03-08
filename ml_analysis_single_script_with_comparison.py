from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


@dataclass(frozen=True)
class Config:
    seed: int = 42
    target: str = "log Kd (L/kg)"
    train_excel: str = "cleaned_with_deltaPZCpH_no planar.xlsx"
    si_excel: str = "SI_Excel_compounds and adsorption info.xlsx"
    si_sheet: str = "Tab. 2ES_Kd_ads._trainandtest."
    repo_zip: str | None = None
    out_dir: str = "ml_manuscript_si_outputs"
    test_size: float = 0.2
    rf_trees: int = 200
    cv_folds: int = 10
    eval_pH_for_delta: float = 7.0


plt.rcParams.update({"figure.dpi": 300, "axes.grid": True})
sns.set_style("whitegrid")


MODEL_SPECS = {
    "RF": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
    "Ridge": Ridge(),
    "SVR": SVR(),
}


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


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_unique_columns(columns) -> list[str]:
    seen = {}
    out = []
    for c in columns:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "positive charges": "positive charges",
        "positive charges ": "positive charges",
        "negative charges": "negative charges",
        "negative charges ": "negative charges",
        "molecular volume  (cm3/mol/100)  ": "molecular volume  (cm3/mol/100)",
    }
    return df.rename(columns=rename)


def ensure_si_excel(base_dir: Path, cfg: Config) -> Path:
    si_path = base_dir / cfg.si_excel
    if si_path.exists():
        return si_path

    zip_candidates = []
    if cfg.repo_zip is not None:
        zip_candidates.append(base_dir / cfg.repo_zip)
    zip_candidates.extend(sorted(base_dir.glob('*.zip')))

    checked_names = []
    for zip_path in zip_candidates:
        if not zip_path.exists() or zip_path in checked_names:
            continue
        checked_names.append(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            target_name = next((name for name in zf.namelist() if name.endswith(cfg.si_excel)), None)
            if target_name is None:
                continue
            zf.extract(target_name, path=base_dir)
            extracted = base_dir / target_name
            extracted.replace(si_path)
            return si_path

    searched = ', '.join(p.name for p in checked_names) if checked_names else 'no zip files found'
    raise FileNotFoundError(
        f"Could not find '{cfg.si_excel}' in the working directory or inside any zip archive next to the script ({searched})."
    )


def load_excel(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    if sheet_name is None:
        return normalize_column_names(pd.read_excel(path))
    return normalize_column_names(pd.read_excel(path, sheet_name=sheet_name))


def axis_limits(a, b, pad_frac: float = 0.05) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mn = float(min(np.nanmin(a), np.nanmin(b)))
    mx = float(max(np.nanmax(a), np.nanmax(b)))
    span = (mx - mn) if mx > mn else 1.0
    pad = span * pad_frac
    return mn - pad, mx + pad


def add_identity_and_rmse(ax, lims: tuple[float, float], rmse: float) -> None:
    lo, hi = lims
    ax.plot([lo, hi], [lo, hi], "--", lw=1.5, color="black")
    ax.plot([lo, hi], [lo + rmse, hi + rmse], ":", lw=1.2, color="black")
    ax.plot([lo, hi], [lo - rmse, hi - rmse], ":", lw=1.2, color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def score_regression(y_true, y_pred) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return r2, rmse, mae


def distance_correlation_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if n != y.shape[0] or n < 2:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    if dcov2_xx <= 0 or dcov2_yy <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2_xy, 0.0)) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy)))


def engineer_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = normalize_column_names(df).copy()
    pos_candidates = [c for c in df.columns if "positive charge" in c.lower()]
    neg_candidates = [c for c in df.columns if "negative charge" in c.lower()]
    if not pos_candidates or not neg_candidates:
        raise RuntimeError("Could not detect positive/negative charge columns.")
    pos_col = pos_candidates[0]
    neg_col = neg_candidates[0]

    df["has_aromatic_ring"] = (df["number of aromatic rings"] >= 1).astype(int)
    df["has_two_aromatics"] = (df["number of aromatic rings"] >= 2).astype(int)
    df["charge_state_anion"] = ((df[pos_col] == 0) & (df[neg_col] == 1)).astype(int)
    df["charge_state_cation"] = ((df[pos_col] == 1) & (df[neg_col] == 0)).astype(int)
    df["charge_state_neutral"] = ((df[pos_col] == 0) & (df[neg_col] == 0)).astype(int)
    df["charge_state_zwitterionic"] = ((df[pos_col] == 1) & (df[neg_col] == 1)).astype(int)
    df = df.drop(columns=[pos_col, neg_col])

    all_features = df.drop(columns=[target, "number of aromatic rings"], errors="ignore").copy()
    all_features.columns = make_unique_columns(all_features.columns)
    y = df[target].astype(float)
    return all_features, y, df


def select_feature_set(X_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    molecular = [c for c in MOLECULAR_FEATURE_ORDER if c in X_all.columns]
    adsorbent = [c for c in ADSORBENT_FEATURE_ORDER if c in X_all.columns]
    if mode == "molecular_only":
        feats = molecular
    elif mode == "molecular_plus_adsorbent":
        feats = molecular + adsorbent
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    if not feats:
        raise RuntimeError(f"No features found for mode '{mode}'.")
    return X_all[feats].copy()


def complete_case_filter(X: pd.DataFrame, y: pd.Series, target: str) -> tuple[pd.DataFrame, pd.Series]:
    merged = pd.concat([X, y.rename(target)], axis=1).replace([np.inf, -np.inf], np.nan)
    before_n = len(merged)
    merged = merged.dropna(axis=0, how="any").copy()
    after_n = len(merged)
    print(f"Complete-case filtering for selected set: dropped {before_n - after_n}, kept {after_n} rows.")
    return merged.drop(columns=[target]), merged[target].astype(float)


def attach_metadata(df_processed: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    meta = df_processed.loc[idx, [
        "number of aromatic rings",
        "has_aromatic_ring",
        "has_two_aromatics",
        "charge_state_anion",
        "charge_state_cation",
        "charge_state_neutral",
        "charge_state_zwitterionic",
    ]].copy()

    def charge_label(row):
        if row["charge_state_anion"] == 1:
            return "Anionic"
        if row["charge_state_cation"] == 1:
            return "Cationic"
        if row["charge_state_neutral"] == 1:
            return "Neutral"
        if row["charge_state_zwitterionic"] == 1:
            return "Zwitterionic"
        return "Unlabeled"

    meta["ChargeClass"] = meta.apply(charge_label, axis=1)
    meta["Aromaticity"] = np.where(meta["number of aromatic rings"] >= 1, "Aromatic ring", "No ring")
    return meta


def plot_si_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    molecular_features = [
        "log D (L/kg)",
        "molecular weight (g/mol)",
        "Log S (mol/L) at pH 7",
        "molecular volume  (cm3/mol/100)",
        "estimated log KOC  (L/kg)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, feat in enumerate(molecular_features):
        axes[i].scatter(df[feat], df["log Kd (L/kg)"], s=24, edgecolor="black", linewidth=0.3)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("log Kd (L/kg)")
        axes[i].set_title(feat)
    axes[-1].axis("off")
    save_fig(out_dir / "Fig_1S_logKd_vs_molecular_properties.png")

    adsorbent_features = ["surface area (m2/g)", "O wt%", "delta_PZCpH"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, feat in zip(axes, adsorbent_features):
        ax.scatter(df[feat], df["log Kd (L/kg)"], s=24, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel("log Kd (L/kg)")
        ax.set_title(feat)
    save_fig(out_dir / "Fig_2S_logKd_vs_adsorbent_properties.png")

    plt.figure(figsize=(5.1, 4.2))
    plt.scatter(df["estimated log KOC  (L/kg)"], df["log Kd (L/kg)"], s=24, edgecolor="black", linewidth=0.3)
    plt.xlabel("estimated log KOC  (L/kg)")
    plt.ylabel("log Kd (L/kg)")
    plt.title("Experimental log Kd vs estimated log KOC")
    save_fig(out_dir / "Fig_3S_logKd_vs_logKOC.png")

    pairs = [
        ("surface area (m2/g)", "O wt%"),
        ("surface area (m2/g)", "delta_PZCpH"),
        ("delta_PZCpH", "O wt%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (xcol, ycol) in zip(axes, pairs):
        ax.scatter(df[xcol], df[ycol], s=24, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{xcol} vs {ycol}")
    save_fig(out_dir / "Fig_4S_intercorrelations_adsorbent_descriptors.png")


def fit_pipeline_and_predict(estimator, X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    model = clone(estimator)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return model, scaler, X_train_s, X_test_s, y_pred


def evaluate_models_for_feature_set(X: pd.DataFrame, y: pd.Series, cfg: Config) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    results = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": {},
    }
    for name, estimator in MODEL_SPECS.items():
        model, scaler, X_train_s, X_test_s, y_pred = fit_pipeline_and_predict(estimator, X_train, X_test, y_train)
        r2, rmse, mae = score_regression(y_test, y_pred)
        results["models"][name] = {
            "model": model,
            "scaler": scaler,
            "X_train_s": X_train_s,
            "X_test_s": X_test_s,
            "y_pred": y_pred,
            "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
        }
        print(f"{name} ({len(X.columns)} features): R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return results


def plot_model_comparison_figure(results: dict, figure_label: str, out_path: Path) -> None:
    y_test = results["y_test"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    for ax, model_name in zip(axes, ["RF", "Ridge", "SVR"]):
        pred = results["models"][model_name]["y_pred"]
        r2 = results["models"][model_name]["metrics"]["R2"]
        rmse = results["models"][model_name]["metrics"]["RMSE"]
        lims = axis_limits(y_test, pred)
        add_identity_and_rmse(ax, lims, rmse)
        ax.scatter(y_test, pred, s=40, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{model_name}\nR²={r2:.2f}, RMSE={rmse:.2f}")
        ax.set_xlabel("Actual log Kd (L/kg)")
        ax.set_ylabel("Predicted log Kd (L/kg)")
    fig.suptitle(figure_label, y=1.02)
    save_fig(out_path)


def plot_individual_model_panels(results: dict, prefix: str, out_dir: Path) -> None:
    y_test = results["y_test"]
    for model_name in ["RF", "Ridge", "SVR"]:
        pred = results["models"][model_name]["y_pred"]
        r2 = results["models"][model_name]["metrics"]["R2"]
        rmse = results["models"][model_name]["metrics"]["RMSE"]
        mae = results["models"][model_name]["metrics"]["MAE"]
        lims = axis_limits(y_test, pred)
        plt.figure(figsize=(5.2, 4.8))
        ax = plt.gca()
        add_identity_and_rmse(ax, lims, rmse)
        ax.scatter(y_test, pred, s=40, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Actual log Kd (L/kg)")
        ax.set_ylabel("Predicted log Kd (L/kg)")
        ax.set_title(f"{model_name}: R²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        save_fig(out_dir / f"{prefix}_{model_name}_predicted_vs_actual.png")


def compute_distance_correlation(X_train_s: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X_train_s.columns:
        rows.append((col, distance_correlation_1d(X_train_s[col].to_numpy(), y_train.to_numpy())))
    return pd.DataFrame(rows, columns=["Feature", "Distance Correlation"]).sort_values(
        "Distance Correlation", ascending=False
    ).reset_index(drop=True)


def compute_mutual_information(X_train_s: pd.DataFrame, y_train: pd.Series, seed: int) -> pd.DataFrame:
    mi = mutual_info_regression(X_train_s.values, y_train.values, random_state=seed)
    return pd.DataFrame({"Feature": X_train_s.columns, "Mutual Information": mi}).sort_values(
        "Mutual Information", ascending=False
    ).reset_index(drop=True)


def plot_main_figure_2(dcor_df: pd.DataFrame, mi_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].barh(dcor_df["Feature"], dcor_df["Distance Correlation"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Distance correlation")
    axes[0].set_title("a")
    axes[1].barh(mi_df["Feature"], mi_df["Mutual Information"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mutual information")
    axes[1].set_title("b")
    save_fig(out_dir / "Fig_2_distance_correlation_and_mutual_information.png")


def plot_main_figure_3c(y_test, y_pred, meta, rmse, r2, mae, out_dir: Path) -> None:
    color_map = {
        "Anionic": "#1f77b4",
        "Neutral": "#7f7f7f",
        "Cationic": "#d62728",
        "Zwitterionic": "#2ca02c",
        "Unlabeled": "#9467bd",
    }
    marker_map = {"Aromatic ring": "^", "No ring": "o"}
    plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index).join(meta)
    lims = axis_limits(plot_df["Actual"], plot_df["Predicted"])
    plt.figure(figsize=(8.8, 7.0))
    ax = plt.gca()
    add_identity_and_rmse(ax, lims, rmse)
    for charge in ["Anionic", "Neutral", "Cationic", "Zwitterionic", "Unlabeled"]:
        for arom in ["Aromatic ring", "No ring"]:
            m = (plot_df["ChargeClass"] == charge) & (plot_df["Aromaticity"] == arom)
            if m.any():
                ax.scatter(
                    plot_df.loc[m, "Actual"],
                    plot_df.loc[m, "Predicted"],
                    s=56,
                    c=color_map[charge],
                    marker=marker_map[arom],
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.95,
                    label=f"{charge}, {arom}",
                )
    ax.set_xlabel("Actual log Kd (L/kg)")
    ax.set_ylabel("Predicted log Kd (L/kg)")
    ax.set_title(f"RF test set: R² = {r2:.2f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_fig(out_dir / "Fig_3c_RF_predicted_vs_actual_colored.png")


def plot_main_figure_3d(model, X_test_s, y_test, y_pred, out_dir: Path):
    Xn = X_test_s.to_numpy()
    tree_preds = np.stack([tree.predict(Xn) for tree in model.estimators_])
    std_devs = np.std(tree_preds, axis=0)
    plt.figure(figsize=(6.4, 5.4))
    sc = plt.scatter(y_test, y_pred, c=std_devs, cmap="viridis", edgecolor="black", linewidth=0.3)
    plt.colorbar(sc, label="Prediction standard deviation")
    lims = axis_limits(y_test, y_pred)
    lo, hi = lims
    plt.plot([lo, hi], [lo, hi], "--", color="black", lw=1.5)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual log Kd (L/kg)")
    plt.ylabel("Predicted log Kd (L/kg)")
    plt.title("RF uncertainty map")
    save_fig(out_dir / "Fig_3d_RF_uncertainty_map.png")
    return std_devs


def run_cross_validation(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=cfg.seed, n_jobs=1)),
    ])
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=1)
    rmse_scores = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=1))
    return pd.DataFrame({"R2": r2_scores, "RMSE": rmse_scores})


def plot_si_figure_7s(cv_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    sns.boxplot(y=cv_df["R2"], ax=axes[0], color="#9ecae1")
    axes[0].set_title("R² across 10 folds")
    axes[0].set_ylabel("R²")
    sns.boxplot(y=cv_df["RMSE"], ax=axes[1], color="#fdd0a2")
    axes[1].set_title("RMSE across 10 folds")
    axes[1].set_ylabel("RMSE (log Kd)")
    save_fig(out_dir / "Fig_7S_RF_10fold_cv.png")


def build_residual_df(y_test, y_pred, meta):
    residuals = y_test.to_numpy() - y_pred
    threshold = float(2 * np.std(residuals))
    res_df = pd.DataFrame(
        {"Actual": y_test.values, "Predicted": y_pred, "Residual": residuals, "AbsError": np.abs(residuals)},
        index=y_test.index,
    ).join(meta)
    res_df["Outlier"] = np.abs(res_df["Residual"]) > threshold
    return res_df, threshold


def plot_si_figure_8s(res_df, threshold, out_dir: Path) -> None:
    plot_df = res_df.reset_index(drop=True).copy()
    plot_df["TestSampleIndex"] = np.arange(1, len(plot_df) + 1)
    plt.figure(figsize=(8.2, 4.3))
    plt.scatter(plot_df["TestSampleIndex"], plot_df["Residual"], s=36, edgecolor="black", linewidth=0.3)
    plt.axhline(0, color="black", lw=1)
    plt.axhline(threshold, color="black", ls="--", lw=1)
    plt.axhline(-threshold, color="black", ls="--", lw=1)
    plt.xlabel("Test sample index")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title("Residuals and outlier band")
    save_fig(out_dir / "Fig_8S_residuals_and_outliers.png")


def plot_main_figure_6(res_df, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sns.boxplot(data=res_df, x="ChargeClass", y="Residual", ax=axes[0])
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set_title("a")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Actual − Predicted log Kd")
    axes[0].tick_params(axis="x", rotation=25)
    sns.boxplot(data=res_df, x="Aromaticity", y="Residual", ax=axes[1])
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set_title("b")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Actual − Predicted log Kd")
    save_fig(out_dir / "Fig_6_residual_boxplots.png")


def plot_si_figure_9s(X_train_s, X_test_s, out_dir: Path):
    cov = EmpiricalCovariance().fit(X_train_s)
    md = cov.mahalanobis(X_test_s)
    plt.figure(figsize=(6.0, 4.2))
    plt.hist(md, bins=20, edgecolor="black")
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Frequency")
    plt.title("Applicability domain")
    save_fig(out_dir / "Fig_9S_applicability_domain_histogram.png")
    return md


def run_shap(model, X_train_s, X_test_s, out_dir: Path):
    if not SHAP_AVAILABLE:
        warnings.warn("SHAP not available; SHAP figures skipped.")
        return None
    explainer = shap.Explainer(model, X_train_s)
    shap_values = explainer(X_test_s, check_additivity=False)
    shap.summary_plot(shap_values, X_test_s, show=False, max_display=20)
    save_fig(out_dir / "Fig_4_SHAP_summary.png")
    return shap_values


def plot_main_figure_5(model, X_test_s, out_dir: Path) -> None:
    features = [
        "molecular volume  (cm3/mol/100)",
        "molecular weight (g/mol)",
        "log D (L/kg)",
        "estimated log KOC  (L/kg)",
        "surface area (m2/g)",
        "O wt%",
        "delta_PZCpH",
        "Log S (mol/L) at pH 7",
    ]
    features = [f for f in features if f in X_test_s.columns]
    fig, ax = plt.subplots(figsize=(14, 9))
    PartialDependenceDisplay.from_estimator(model, X_test_s, features=features, feature_names=list(X_test_s.columns), ax=ax)
    save_fig(out_dir / "Fig_5_partial_dependence_selected_features.png")


def compute_adsorbent_influence_figures(shap_values, X_test_s, res_df, out_dir: Path):
    if shap_values is None:
        return
    adsorbent_features = ["surface area (m2/g)", "O wt%", "delta_PZCpH"]
    if any(f not in X_test_s.columns for f in adsorbent_features):
        return
    shap_matrix = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    abs_shap = np.abs(shap_matrix)
    infl_df = res_df.copy()
    for f in adsorbent_features:
        infl_df[f"ABS_SHAP__{f}"] = abs_shap[:, list(X_test_s.columns).index(f)]
    infl_df["ABS_SHAP__AdsorbentSum"] = infl_df[[f"ABS_SHAP__{f}" for f in adsorbent_features]].sum(axis=1)
    infl_df["charge_x_aromatic"] = infl_df["ChargeClass"].astype(str) + " × " + infl_df["Aromaticity"].astype(str)
    ordered = infl_df.groupby("charge_x_aromatic")["ABS_SHAP__AdsorbentSum"].mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sns.violinplot(data=infl_df, x="charge_x_aromatic", y="ABS_SHAP__AdsorbentSum", order=ordered, ax=axes[0], inner="box", cut=0)
    axes[0].set_title("a")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Summed absolute SHAP of adsorbent features")
    axes[0].tick_params(axis="x", rotation=30)
    stacked = infl_df.groupby("charge_x_aromatic")[[f"ABS_SHAP__{f}" for f in adsorbent_features]].mean().loc[ordered]
    bottom = np.zeros(len(stacked))
    for col in stacked.columns:
        axes[1].bar(stacked.index, stacked[col].values, bottom=bottom, label=col.replace("ABS_SHAP__", ""))
        bottom += stacked[col].values
    axes[1].set_title("b")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean absolute SHAP")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Feature")
    save_fig(out_dir / "Fig_7_adsorbent_influence_by_charge_x_aromaticity.png")


def load_external_evaluation_dataset(cfg: Config, training_n_rows: int, base_dir: Path) -> pd.DataFrame:
    combined = load_excel(ensure_si_excel(base_dir, cfg), sheet_name=cfg.si_sheet)
    eval_df = combined.iloc[training_n_rows:].copy()
    eval_df["delta_PZCpH"] = eval_df["PZC"].astype(float) - cfg.eval_pH_for_delta
    eval_df = eval_df.rename(columns={"log Kd": cfg.target})
    return eval_df


def prepare_external_features(eval_df: pd.DataFrame, train_columns: list[str], target: str):
    X_eval_all, y_eval, eval_processed = engineer_features(eval_df, target)
    X_eval = X_eval_all[[c for c in train_columns if c in X_eval_all.columns]].copy()
    merged = pd.concat([X_eval, y_eval.rename(target)], axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    X_eval = merged.drop(columns=[target]).copy()
    y_eval = merged[target].astype(float)
    missing = [c for c in train_columns if c not in X_eval.columns]
    if missing:
        raise RuntimeError(f"External evaluation data missing columns: {missing}")
    X_eval = X_eval[train_columns]
    return X_eval, y_eval, eval_processed.loc[X_eval.index].copy()


def net_charge_label(meta: pd.DataFrame) -> pd.Series:
    out = pd.Series("neutral", index=meta.index)
    out.loc[meta["charge_state_anion"] == 1] = "negative"
    out.loc[meta["charge_state_cation"] == 1] = "positive"
    out.loc[meta["charge_state_zwitterionic"] == 1] = "zwitterionic"
    return out


def plot_main_figure_8(model, scaler, X_eval, y_eval, eval_meta, out_dir: Path):
    X_eval_s = pd.DataFrame(scaler.transform(X_eval), index=X_eval.index, columns=X_eval.columns)
    y_eval_pred = model.predict(X_eval_s)
    r2, rmse, mae = score_regression(y_eval, y_eval_pred)
    eval_res = pd.DataFrame({"Actual": y_eval.values, "Predicted": y_eval_pred}, index=y_eval.index).join(eval_meta)
    eval_res["NetCharge"] = net_charge_label(eval_meta)
    eval_res["Aromaticity"] = np.where(eval_meta["number of aromatic rings"] >= 1, "Aromatic ring", "No ring")
    color_map = {"negative": "#1f77b4", "neutral": "#7f7f7f", "positive": "#d62728", "zwitterionic": "#2ca02c"}
    marker_map = {"Aromatic ring": "^", "No ring": "o"}
    lims = axis_limits(eval_res["Actual"], eval_res["Predicted"])
    plt.figure(figsize=(8.2, 6.6))
    ax = plt.gca()
    add_identity_and_rmse(ax, lims, rmse)
    for charge in ["negative", "neutral", "positive", "zwitterionic"]:
        for arom in ["Aromatic ring", "No ring"]:
            m = (eval_res["NetCharge"] == charge) & (eval_res["Aromaticity"] == arom)
            if m.any():
                ax.scatter(
                    eval_res.loc[m, "Actual"],
                    eval_res.loc[m, "Predicted"],
                    s=58,
                    c=color_map[charge],
                    marker=marker_map[arom],
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.95,
                    label=f"{charge}, {arom}",
                )
    ax.set_xlabel("Actual log Kd (L/kg)")
    ax.set_ylabel("Predicted log Kd (L/kg)")
    ax.set_title(f"Independent evaluation: R² = {r2:.2f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_fig(out_dir / "Fig_8_independent_evaluation.png")
    eval_res["Residual"] = eval_res["Actual"] - eval_res["Predicted"]
    eval_res["AbsError"] = np.abs(eval_res["Residual"])
    return eval_res


def main() -> None:
    cfg = Config()
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_excel(base_dir / cfg.train_excel)
    plot_si_correlations(df_raw, out_dir)

    X_all, y_all, df_processed = engineer_features(df_raw, cfg.target)

    X_mol, y_mol = complete_case_filter(select_feature_set(X_all, "molecular_only"), y_all, cfg.target)
    X_full, y_full = complete_case_filter(select_feature_set(X_all, "molecular_plus_adsorbent"), y_all, cfg.target)

    mol_results = evaluate_models_for_feature_set(X_mol, y_mol, cfg)
    full_results = evaluate_models_for_feature_set(X_full, y_full, cfg)

    plot_model_comparison_figure(
        mol_results,
        "Figure 3a: molecular properties only",
        out_dir / "Fig_3a_model_comparison_molecular_only.png",
    )
    plot_model_comparison_figure(
        full_results,
        "Figure 3b: molecular + adsorbent properties",
        out_dir / "Fig_3b_model_comparison_molecular_plus_adsorbent.png",
    )

    plot_individual_model_panels(full_results, "Fig_5S_full_features", out_dir)
    plot_individual_model_panels(mol_results, "Fig_6S_molecular_only", out_dir)

    rf_full = full_results["models"]["RF"]
    y_test_rf = full_results["y_test"]
    test_meta = attach_metadata(df_processed, y_test_rf.index)

    dcor_df = compute_distance_correlation(rf_full["X_train_s"], full_results["y_train"])
    mi_df = compute_mutual_information(rf_full["X_train_s"], full_results["y_train"], cfg.seed)
    dcor_df.to_csv(out_dir / "distance_correlation_results.csv", index=False)
    mi_df.to_csv(out_dir / "mutual_information_results.csv", index=False)
    plot_main_figure_2(dcor_df, mi_df, out_dir)

    r2 = rf_full["metrics"]["R2"]
    rmse = rf_full["metrics"]["RMSE"]
    mae = rf_full["metrics"]["MAE"]
    plot_main_figure_3c(y_test_rf, rf_full["y_pred"], test_meta, rmse, r2, mae, out_dir)
    plot_main_figure_3d(rf_full["model"], rf_full["X_test_s"], y_test_rf, rf_full["y_pred"], out_dir)

    cv_df = run_cross_validation(X_full, y_full, cfg)
    cv_df.to_csv(out_dir / "rf_10fold_cv_scores.csv", index=False)
    plot_si_figure_7s(cv_df, out_dir)

    res_df, threshold = build_residual_df(y_test_rf, rf_full["y_pred"], test_meta)
    res_df.to_csv(out_dir / "test_residuals_with_meta.csv", index=True)
    plot_si_figure_8s(res_df, threshold, out_dir)
    plot_main_figure_6(res_df, out_dir)
    plot_si_figure_9s(rf_full["X_train_s"], rf_full["X_test_s"], out_dir)

    shap_values = run_shap(rf_full["model"], rf_full["X_train_s"], rf_full["X_test_s"], out_dir)
    plot_main_figure_5(rf_full["model"], rf_full["X_test_s"], out_dir)
    compute_adsorbent_influence_figures(shap_values, rf_full["X_test_s"], res_df, out_dir)

    eval_raw = load_external_evaluation_dataset(cfg, training_n_rows=len(df_raw), base_dir=base_dir)
    X_eval, y_eval, eval_processed = prepare_external_features(eval_raw, list(X_full.columns), cfg.target)
    eval_meta = attach_metadata(eval_processed, X_eval.index)
    eval_res = plot_main_figure_8(rf_full["model"], rf_full["scaler"], X_eval, y_eval, eval_meta, out_dir)
    eval_res.to_csv(out_dir / "independent_evaluation_results.csv", index=True)

    summary_rows = []
    for feature_set_name, results in [("molecular_only", mol_results), ("molecular_plus_adsorbent", full_results)]:
        for model_name in ["RF", "Ridge", "SVR"]:
            metrics = results["models"][model_name]["metrics"]
            summary_rows.append({
                "feature_set": feature_set_name,
                "model": model_name,
                **metrics,
                "n_samples": len(results["X"]),
                "n_features": results["X"].shape[1],
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "model_comparison_summary.csv", index=False)

    print("Finished successfully.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
