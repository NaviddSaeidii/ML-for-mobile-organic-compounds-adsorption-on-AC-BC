#!/usr/bin/env python3
"""
Supplementary revision analyses for:
"Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning"

This script reproduces the additional analyses added during revision:
1) MBE for the random held-out test set and independent evaluation set
2) Standard 10-fold RF cross-validation
3) Compound-grouped 10-fold RF cross-validation using GroupKFold
4) PFAS vs. non-PFAS performance metrics
5) Independent-set applicability-domain analysis using Mahalanobis distance
6) Limited SVR tuning sensitivity check
7) Optional SHAP dependence plot for delta_PZCpH, if shap is installed

Place this script in the same folder as:
- merged_train_external.csv
- SI_Excel_compounds and adsorption info.xlsx

Run:
    python revision_additional_analyses.py

Optional:
    python revision_additional_analyses.py --combined_csv merged_train_external.csv --metadata_excel "SI_Excel_compounds and adsorption info.xlsx"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2

from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass(frozen=True)
class Config:
    seed: int = 42
    target: str = "log Kd (L/kg)"
    combined_csv: str = "merged_train_external.csv"
    metadata_excel: str = "SI_Excel_compounds and adsorption info.xlsx"
    metadata_sheet: str = "Tab. 2ES_Kd_ads._trainandtest."
    train_rows: int = 509
    test_size: float = 0.2
    rf_trees: int = 200
    cv_folds: int = 10
    out_dir: str = "revision_additional_analyses_outputs"


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


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "positive charges ": "positive charges",
        "negative charges ": "negative charges",
        "molecular volume  (cm3/mol/100)  ": "molecular volume  (cm3/mol/100)",
        "log Kd": "log Kd (L/kg)",
    }
    return df.rename(columns=rename)



def normalize_compound_label(value: object) -> str:
    """Normalize compound identity for compound-grouped validation.

    This avoids treating the same compound as different groups only because of
    capitalization or extra whitespace differences in the SI table.
    """
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def load_combined_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {path.resolve()}")
    return normalize_column_names(pd.read_csv(path))


def load_metadata_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata Excel file not found: {path.resolve()}\n"
            "Compound labels are required for GroupKFold and PFAS/non-PFAS metrics. "
            "Place the Excel SI file next to this script or pass --metadata_excel."
        )
    return normalize_column_names(pd.read_excel(path, sheet_name=sheet_name))


def attach_compound_labels(combined: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Attach compound labels to the combined dataframe.

    The merged CSV used by the model contains only numeric features. The Excel SI contains
    the compound names in the same row order. If the combined CSV already contains a
    'compound' column, that column is used directly.
    """
    out = combined.copy()
    if "compound" in out.columns:
        out["compound"] = out["compound"].astype(str)
        out["compound_group"] = out["compound"].map(normalize_compound_label)
        return out

    if metadata is None or "compound" not in metadata.columns:
        raise RuntimeError(
            "No compound column found in the combined CSV or metadata Excel file. "
            "Compound labels are required for GroupKFold and PFAS/non-PFAS metrics."
        )

    if len(metadata) != len(out):
        raise RuntimeError(
            f"Metadata row count ({len(metadata)}) does not match combined CSV row count ({len(out)}). "
            "Cannot safely attach compound labels."
        )

    out["compound"] = metadata["compound"].astype(str).values
    out["compound_group"] = out["compound"].map(normalize_compound_label)
    return out


def find_charge_columns(df: pd.DataFrame) -> tuple[str, str]:
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
            "Could not detect positive/negative charge columns. Expected columns like "
            "'positive charges' and 'negative charges'."
        )

    return positive_candidates[0], negative_candidates[0]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the binary features used in the manuscript comparison script."""
    work = normalize_column_names(df)

    positive_col, negative_col = find_charge_columns(work)

    work["number of aromatic rings"] = pd.to_numeric(
        work["number of aromatic rings"], errors="coerce"
    )
    positive = pd.to_numeric(work[positive_col], errors="coerce")
    negative = pd.to_numeric(work[negative_col], errors="coerce")

    work["has_aromatic_ring"] = (work["number of aromatic rings"] >= 1).astype(int)
    work["has_two_aromatics"] = (work["number of aromatic rings"] >= 2).astype(int)

    # Use the same binary charge-state logic as the manuscript model script.
    work["charge_state_anion"] = ((positive == 0) & (negative == 1)).astype(int)
    work["charge_state_cation"] = ((positive == 1) & (negative == 0)).astype(int)
    work["charge_state_neutral"] = ((positive == 0) & (negative == 0)).astype(int)
    work["charge_state_zwitterionic"] = ((positive == 1) & (negative == 1)).astype(int)

    return work


def complete_case_xy(work: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    missing = [c for c in features + [target] if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = work[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work[target], errors="coerce")

    merged = (
        pd.concat([X, y.rename(target)], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )

    X_cc = merged[features].copy()
    y_cc = pd.to_numeric(merged[target], errors="coerce").copy()
    meta_cc = work.loc[X_cc.index].copy()

    return X_cc, y_cc, meta_cc


def rf_pipeline(cfg: Config) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=cfg.seed, n_jobs=1)),
    ])


def score_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return metrics with MBE defined as predicted - actual."""
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_arr)),
        "R2": float(r2_score(y_arr, pred_arr)) if len(y_arr) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_arr, pred_arr))),
        "MAE": float(mean_absolute_error(y_arr, pred_arr)),
        "MBE_pred_minus_actual": float(np.mean(pred_arr - y_arr)),
    }


def predict_random_holdout(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, cfg: Config):
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=cfg.test_size, random_state=cfg.seed
    )

    pipe = rf_pipeline(cfg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred,
        "Residual_actual_minus_predicted": y_test.values - pred,
        "Error_predicted_minus_actual": pred - y_test.values,
    }, index=y_test.index).join(meta_test[["compound"]], how="left")

    metrics = score_predictions(y_test, pred)
    return metrics, pred_df, (X_train, X_test, y_train, y_test, meta_train, meta_test, pipe)


def run_standard_kfold_cv(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        pipe = rf_pipeline(cfg)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        metrics = score_predictions(y.iloc[test_idx], pred)
        rows.append({
            "fold": fold,
            "n_test": int(len(test_idx)),
            **{k: v for k, v in metrics.items() if k != "n"},
        })

    return pd.DataFrame(rows)


def run_groupkfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: Config) -> pd.DataFrame:
    if groups.isna().any():
        raise RuntimeError("GroupKFold cannot run because some compound labels are missing.")

    n_groups = groups.nunique()
    if n_groups < cfg.cv_folds:
        raise RuntimeError(f"GroupKFold requested {cfg.cv_folds} folds, but only {n_groups} groups are available.")

    rows = []
    gkf = GroupKFold(n_splits=cfg.cv_folds)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        train_groups = set(pd.Series(groups.iloc[train_idx]).astype(str))
        test_groups = set(pd.Series(groups.iloc[test_idx]).astype(str))
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(
                f"Group leakage detected in fold {fold}: {sorted(overlap)[:10]}"
            )

        pipe = rf_pipeline(cfg)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        metrics = score_predictions(y.iloc[test_idx], pred)

        heldout_compounds = sorted(pd.Series(groups.iloc[test_idx]).astype(str).unique())
        rows.append({
            "fold": fold,
            "n_test": int(len(test_idx)),
            "n_heldout_compounds": int(len(heldout_compounds)),
            "heldout_compounds": "; ".join(heldout_compounds),
            **{k: v for k, v in metrics.items() if k != "n"},
        })

    return pd.DataFrame(rows)


def summarize_cv(cv_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out = {}
    for metric in ["R2", "RMSE", "MAE", "MBE_pred_minus_actual"]:
        out[f"{prefix}_{metric}_mean"] = float(cv_df[metric].mean())
        out[f"{prefix}_{metric}_median"] = float(cv_df[metric].median())
        out[f"{prefix}_{metric}_min"] = float(cv_df[metric].min())
        out[f"{prefix}_{metric}_max"] = float(cv_df[metric].max())
    return out


def is_pfas_compound(name: str) -> bool:
    """Operational PFAS label used for the subset analysis.

    This reproduces the manuscript grouping for the uploaded dataset:
    PFAS are identified by names/abbreviations starting with 'PF' or containing 'perfluoro'.
    """
    s = str(name).strip().upper()
    return s.startswith("PF") or ("PERFLUORO" in s)


def add_subset_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "compound" not in out.columns:
        raise RuntimeError("PFAS/non-PFAS analysis requires a 'compound' column.")
    out["is_PFAS"] = out["compound"].map(is_pfas_compound)
    out["subset"] = np.where(out["is_PFAS"], "PFAS", "non-PFAS")
    return out


def subset_metrics(pred_df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    rows = []
    pred_df = add_subset_columns(pred_df)
    for subset_name, part in pred_df.groupby("subset", sort=False):
        metrics = score_predictions(part["Actual"], part["Predicted"])
        rows.append({
            "Dataset/subset": f"{dataset_label}, {subset_name}",
            **metrics,
        })
    return pd.DataFrame(rows)


def train_final_rf_and_predict_external(
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_meta: pd.DataFrame,
    cfg: Config,
) -> tuple[dict[str, float], pd.DataFrame, Pipeline]:
    pipe = rf_pipeline(cfg)
    pipe.fit(X_train_all, y_train_all)
    pred = pipe.predict(X_eval)

    pred_df = pd.DataFrame({
        "Actual": y_eval.values,
        "Predicted": pred,
        "Residual_actual_minus_predicted": y_eval.values - pred,
        "Error_predicted_minus_actual": pred - y_eval.values,
    }, index=y_eval.index).join(eval_meta[["compound"]], how="left")

    metrics = score_predictions(y_eval, pred)
    return metrics, pred_df, pipe


def run_external_applicability_domain(
    X_train_all: pd.DataFrame,
    X_eval: pd.DataFrame,
    eval_meta: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mahalanobis distance in standardized model-feature space."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_all), index=X_train_all.index, columns=X_train_all.columns
    )
    X_eval_s = pd.DataFrame(
        scaler.transform(X_eval), index=X_eval.index, columns=X_eval.columns
    )

    cov = EmpiricalCovariance().fit(X_train_s)
    md_squared = cov.mahalanobis(X_eval_s)
    md = np.sqrt(md_squared)

    threshold_squared = float(chi2.ppf(0.975, df=X_train_all.shape[1]))
    threshold = float(np.sqrt(threshold_squared))

    ad_df = pd.DataFrame({
        "compound": eval_meta["compound"].astype(str).values,
        "surface area (m2/g)": X_eval["surface area (m2/g)"].values,
        "Mahalanobis_D": md,
        "Mahalanobis_D_squared": md_squared,
        "threshold_D_97p5": threshold,
        "threshold_D_squared_97p5": threshold_squared,
        "outside_AD_97p5": md > threshold,
    }, index=X_eval.index)

    outside = ad_df.loc[ad_df["outside_AD_97p5"]].copy()

    # Grouped version for direct manuscript/SI table checking.
    grouped_rows = []
    for compound, part in outside.groupby("compound", sort=False):
        ssa_str = ", ".join(f"{v:.0f}" for v in part["surface area (m2/g)"].values)
        md_str = ", ".join(f"{v:.2f}" for v in part["Mahalanobis_D"].values)
        if str(compound).strip().lower() == "gabapentin":
            reason = "Molecular/speciation region at edge of training distribution"
        else:
            reason = "Very low SSA relative to training range"

        compound_label = str(compound)
        if compound_label in {"Ibuprofen", "Naproxen", "Methyl paraben"}:
            compound_label = f"{compound_label} (BC)"

        grouped_rows.append({
            "Independent points outside applicability domain": compound_label,
            "Surface area (m2/g)": ssa_str,
            "Mahalanobis D": md_str,
            "Reason for caution": reason,
        })

    outside_grouped = pd.DataFrame(grouped_rows)
    return ad_df, outside_grouped


def run_svr_tuning_check(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )

    default_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR()),
    ])
    default_pipe.fit(X_train, y_train)
    default_pred = default_pipe.predict(X_test)
    default_metrics = score_predictions(y_test, default_pred)

    # Limited sensitivity grid; intentionally small to avoid extensive optimization.
    param_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "svr__epsilon": [0.01, 0.1, 0.2, 0.5],
    }

    tuned_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR()),
    ])

    search = GridSearchCV(
        tuned_pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    tuned_pred = search.predict(X_test)
    tuned_metrics = score_predictions(y_test, tuned_pred)

    rows = [
        {
            "model": "SVR default",
            "best_params": "scikit-learn defaults",
            **default_metrics,
        },
        {
            "model": "SVR limited grid search",
            "best_params": str(search.best_params_),
            "best_cv_neg_RMSE": float(search.best_score_),
            **tuned_metrics,
        },
    ]
    return pd.DataFrame(rows)


def make_table_s2_summary(
    random_metrics: dict[str, float],
    standard_cv: pd.DataFrame,
    group_cv: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Validation setting": "Random held-out test set",
            "n / folds": f"{random_metrics['n']} test points",
            "R2": f"{random_metrics['R2']:.2f}",
            "RMSE": f"{random_metrics['RMSE']:.2f}",
            "MBE": f"{random_metrics['MBE_pred_minus_actual']:+.2f}",
        },
        {
            "Validation setting": "Standard 10-fold CV",
            "n / folds": "10 folds",
            "R2": f"median = {standard_cv['R2'].median():.2f}; mean = {standard_cv['R2'].mean():.2f}",
            "RMSE": f"median = {standard_cv['RMSE'].median():.2f}; mean = {standard_cv['RMSE'].mean():.2f}",
            "MBE": f"median = {standard_cv['MBE_pred_minus_actual'].median():+.2f}; mean = {standard_cv['MBE_pred_minus_actual'].mean():+.2f}",
        },
        {
            "Validation setting": "Compound-grouped 10-fold CV",
            "n / folds": "10 folds",
            "R2": f"median = {group_cv['R2'].median():.2f}; mean = {group_cv['R2'].mean():.2f}",
            "RMSE": f"median = {group_cv['RMSE'].median():.2f}; mean = {group_cv['RMSE'].mean():.2f}",
            "MBE": f"median = {group_cv['MBE_pred_minus_actual'].median():+.2f}; mean = {group_cv['MBE_pred_minus_actual'].mean():+.2f}",
        },
    ])


def try_make_shap_delta_plot(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Optional Fig. S10-style SHAP dependence plot.

    If shap is not installed, this analysis is skipped and the main reproducibility
    outputs are still generated. The RF model is explained in the standardized
    feature space using the scaled training set as the SHAP background and the
    scaled random held-out test set as the explained set.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:
        warnings.warn(f"shap is not installed; skipping SHAP dependence plot. Details: {exc}")
        return

    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["rf"]

    X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    explainer = shap.Explainer(model, X_train_s)
    shap_values = explainer(X_test_s, check_additivity=False)

    shap_matrix = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    delta_idx = list(X_test_s.columns).index("delta_PZCpH")

    def charge_label(row):
        if row["charge_state_anion"] == 1:
            return "anionic"
        if row["charge_state_cation"] == 1:
            return "cationic"
        if row["charge_state_zwitterionic"] == 1:
            return "zwitterionic"
        return "neutral"

    plot_df = pd.DataFrame({
        "delta_PZCpH": X_test["delta_PZCpH"].values,
        "surface area (m2/g)": X_test["surface area (m2/g)"].values,
        "SHAP_delta_PZCpH": shap_matrix[:, delta_idx],
    }, index=X_test.index)
    plot_df["charge class"] = meta_test.apply(charge_label, axis=1).values
    plot_df.to_csv(out_dir / "Fig_S10_SHAP_delta_PZCpH_dependence_plot_data.csv", index=True)

    marker_map = {
        "anionic": "o",
        "neutral": "s",
        "cationic": "^",
        "zwitterionic": "D",
    }

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })

    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=300)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.85", linewidth=0.8, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(0, color="black", linewidth=0.9, linestyle=":", zorder=1)

    sc_for_colorbar = None
    for charge, marker in marker_map.items():
        subset = plot_df.loc[plot_df["charge class"] == charge]
        if subset.empty:
            continue
        sc = ax.scatter(
            subset["delta_PZCpH"],
            subset["SHAP_delta_PZCpH"],
            c=subset["surface area (m2/g)"],
            marker=marker,
            s=72,
            edgecolors="black",
            linewidths=0.45,
            alpha=1.0,
            label=charge,
            cmap="viridis",
            zorder=3,
        )
        sc_for_colorbar = sc

    ax.set_xlabel("delta_PZCpH (PZC − pH)")
    ax.set_ylabel("SHAP value for delta_PZCpH")

    if sc_for_colorbar is not None:
        cbar = fig.colorbar(sc_for_colorbar, ax=ax)
        cbar.set_label("Surface area (m²/g)")

    ax.legend(title="Charge class", loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_S10_SHAP_delta_PZCpH_dependence.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "Fig_S10_SHAP_delta_PZCpH_dependence.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    path: Path,
    random_metrics: dict[str, float],
    external_metrics: dict[str, float],
    standard_cv: pd.DataFrame,
    group_cv: pd.DataFrame,
    svr_df: pd.DataFrame,
    ad_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Revision additional analyses summary")
    lines.append("=" * 45)
    lines.append("")
    lines.append("MBE is defined as predicted - actual log Kd.")
    lines.append("")
    lines.append(
        f"Random held-out RF: n={random_metrics['n']}, "
        f"R2={random_metrics['R2']:.3f}, RMSE={random_metrics['RMSE']:.3f}, "
        f"MAE={random_metrics['MAE']:.3f}, MBE={random_metrics['MBE_pred_minus_actual']:+.3f}"
    )
    lines.append(
        f"Independent evaluation RF: n={external_metrics['n']}, "
        f"R2={external_metrics['R2']:.3f}, RMSE={external_metrics['RMSE']:.3f}, "
        f"MAE={external_metrics['MAE']:.3f}, MBE={external_metrics['MBE_pred_minus_actual']:+.3f}"
    )
    lines.append("")
    lines.append(
        f"Standard 10-fold CV: median R2={standard_cv['R2'].median():.3f}, "
        f"median RMSE={standard_cv['RMSE'].median():.3f}, "
        f"mean R2={standard_cv['R2'].mean():.3f}, mean RMSE={standard_cv['RMSE'].mean():.3f}"
    )
    lines.append(
        f"Compound GroupKFold: median R2={group_cv['R2'].median():.3f}, "
        f"median RMSE={group_cv['RMSE'].median():.3f}, "
        f"mean R2={group_cv['R2'].mean():.3f}, mean RMSE={group_cv['RMSE'].mean():.3f}, "
        f"mean MBE={group_cv['MBE_pred_minus_actual'].mean():+.3f}"
    )
    lines.append("")
    lines.append("SVR tuning check:")
    for _, row in svr_df.iterrows():
        lines.append(
            f"- {row['model']}: R2={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
            f"MBE={row['MBE_pred_minus_actual']:+.3f}, params={row['best_params']}"
        )
    lines.append("")
    outside_n = int(ad_df["outside_AD_97p5"].sum())
    lines.append(
        f"Independent applicability domain: {outside_n} of {len(ad_df)} points outside "
        "the 97.5% Mahalanobis-distance threshold."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revision additional analyses for the ML PMT adsorption manuscript.")
    parser.add_argument("--combined_csv", default=Config.combined_csv)
    parser.add_argument("--metadata_excel", default=Config.metadata_excel)
    parser.add_argument("--metadata_sheet", default=Config.metadata_sheet)
    parser.add_argument("--out_dir", default=Config.out_dir)
    parser.add_argument("--train_rows", type=int, default=Config.train_rows)
    parser.add_argument("--skip_shap", action="store_true", help="Skip optional SHAP dependence plot.")
    args = parser.parse_args()

    cfg = Config(
        combined_csv=args.combined_csv,
        metadata_excel=args.metadata_excel,
        metadata_sheet=args.metadata_sheet,
        out_dir=args.out_dir,
        train_rows=args.train_rows,
    )

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined_csv(base_dir / cfg.combined_csv)
    metadata = load_metadata_excel(base_dir / cfg.metadata_excel, cfg.metadata_sheet)
    combined = attach_compound_labels(combined, metadata)
    work_all = engineer_features(combined)

    train_work = work_all.iloc[:cfg.train_rows].copy()
    eval_work = work_all.iloc[cfg.train_rows:].copy()

    X_train_all, y_train_all, meta_train_all = complete_case_xy(train_work, FEATURE_ORDER, cfg.target)
    X_eval, y_eval, meta_eval = complete_case_xy(eval_work, FEATURE_ORDER, cfg.target)

    # Random held-out RF and MBE
    random_metrics, random_pred_df, split_objects = predict_random_holdout(
        X_train_all, y_train_all, meta_train_all, cfg
    )
    X_train, X_test, y_train, y_test, meta_train, meta_test, random_rf_pipe = split_objects
    random_pred_df.to_csv(out_dir / "random_holdout_predictions_with_metadata.csv", index=True)

    # CV analyses
    standard_cv = run_standard_kfold_cv(X_train_all, y_train_all, cfg)
    # Use normalized compound identities for GroupKFold.
    # This keeps capitalization/whitespace variants of the same compound in the same fold.
    groups = meta_train_all["compound_group"].astype(str)
    group_cv = run_groupkfold_cv(X_train_all, y_train_all, groups, cfg)

    normalization_map = (
        meta_train_all[["compound", "compound_group"]]
        .drop_duplicates()
        .sort_values(["compound_group", "compound"])
    )
    normalization_map.to_csv(out_dir / "compound_group_normalization_map.csv", index=False)

    standard_cv.to_csv(out_dir / "standard_10fold_cv_fold_metrics.csv", index=False)
    group_cv.to_csv(out_dir / "compound_groupkfold_10fold_fold_metrics.csv", index=False)

    table_s2 = make_table_s2_summary(random_metrics, standard_cv, group_cv)
    table_s2.to_csv(out_dir / "Table_S2_validation_strategy_summary.csv", index=False)

    # External RF evaluation
    external_metrics, external_pred_df, final_rf_pipe = train_final_rf_and_predict_external(
        X_train_all, y_train_all, X_eval, y_eval, meta_eval, cfg
    )
    external_pred_df.to_csv(out_dir / "independent_evaluation_predictions_with_metadata.csv", index=True)

    # PFAS/non-PFAS subset metrics
    random_subset = subset_metrics(random_pred_df, "Random test set")
    external_subset = subset_metrics(external_pred_df, "Independent set")
    table_s3 = pd.concat([random_subset, external_subset], ignore_index=True)
    table_s3.to_csv(out_dir / "Table_S3_PFAS_nonPFAS_metrics.csv", index=False)

    # Overall metrics file
    overall = pd.DataFrame([
        {"dataset": "Random held-out test set", **random_metrics},
        {"dataset": "Independent evaluation set", **external_metrics},
    ])
    overall.to_csv(out_dir / "overall_RF_metrics_with_MBE.csv", index=False)

    # Independent AD analysis
    ad_df, ad_grouped = run_external_applicability_domain(X_train_all, X_eval, meta_eval, cfg)
    ad_df.to_csv(out_dir / "independent_evaluation_applicability_domain_all_points.csv", index=True)
    ad_grouped.to_csv(out_dir / "Table_S6_independent_AD_outside_grouped.csv", index=False)

    # SVR tuning check
    svr_df = run_svr_tuning_check(X_train_all, y_train_all, cfg)
    svr_df.to_csv(out_dir / "SVR_limited_tuning_sensitivity_check.csv", index=False)

    # Optional SHAP dependence plot for Fig. S10
    if not args.skip_shap:
        try_make_shap_delta_plot(random_rf_pipe, X_train, X_test, meta_test, out_dir)

    write_summary_text(
        out_dir / "revision_additional_analyses_summary.txt",
        random_metrics=random_metrics,
        external_metrics=external_metrics,
        standard_cv=standard_cv,
        group_cv=group_cv,
        svr_df=svr_df,
        ad_df=ad_df,
    )

    print("Revision additional analyses completed.")
    print(f"Output directory: {out_dir.resolve()}")
    print("")
    print(f"Random held-out RF: R2={random_metrics['R2']:.3f}, RMSE={random_metrics['RMSE']:.3f}, MBE={random_metrics['MBE_pred_minus_actual']:+.3f}")
    print(f"Independent RF: R2={external_metrics['R2']:.3f}, RMSE={external_metrics['RMSE']:.3f}, MBE={external_metrics['MBE_pred_minus_actual']:+.3f}")
    print(f"GroupKFold: median R2={group_cv['R2'].median():.3f}, median RMSE={group_cv['RMSE'].median():.3f}, mean MBE={group_cv['MBE_pred_minus_actual'].mean():+.3f}")
    print(f"External AD: {int(ad_df['outside_AD_97p5'].sum())} / {len(ad_df)} outside 97.5% threshold")
    print("SVR check:")
    for _, row in svr_df.iterrows():
        print(f"  {row['model']}: R2={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, params={row['best_params']}")


if __name__ == "__main__":
    main()
