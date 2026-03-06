from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import dcor
except ImportError as exc:
    raise ImportError(
        "The 'dcor' package is required for the distance-correlation analysis. "
        "Install it with: pip install dcor"
    ) from exc

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


@dataclass(frozen=True)
class Config:
    seed: int = 42
    target: str = "log Kd (L/kg)"
    excel_name: str = "cleaned_with_deltaPZCpH_no planar.xlsx"
    out_dir: str = "outputs"
    test_size: float = 0.2
    rf_trees: int = 200
    y_randomization_runs: int = 10
    cv_folds_y_randomization: int = 5
    cv_folds_learning_curve: int = 5
    cv_folds_model_check: int = 10


plt.rcParams.update({"figure.dpi": 300, "axes.grid": True})


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_unique_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        if column not in seen:
            seen[column] = 1
            unique_columns.append(column)
        else:
            seen[column] += 1
            unique_columns.append(f"{column}.{seen[column]}")
    return unique_columns


def load_dataset(cfg: Config) -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / cfg.excel_name
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    df = pd.read_excel(file_path)
    print("Columns found in Excel:", df.columns.tolist())

    if cfg.target not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target}' not found in Excel file.")

    return df


def build_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()

    if "number of aromatic rings" not in df.columns:
        raise RuntimeError("'number of aromatic rings' column not found.")

    df["has_aromatic_ring"] = (df["number of aromatic rings"] >= 1).astype(int)
    df["has_two_aromatics"] = (df["number of aromatic rings"] >= 2).astype(int)

    positive_candidates = [col for col in df.columns if "positive charge" in col.lower()]
    negative_candidates = [col for col in df.columns if "negative charge" in col.lower()]
    if not positive_candidates or not negative_candidates:
        raise RuntimeError("Could not detect the positive and negative charge columns automatically.")

    positive_col = positive_candidates[0]
    negative_col = negative_candidates[0]

    df["charge_state_anion"] = ((df[positive_col] == 0) & (df[negative_col] == 1)).astype(int)
    df["charge_state_cation"] = ((df[positive_col] == 1) & (df[negative_col] == 0)).astype(int)
    df["charge_state_neutral"] = ((df[positive_col] == 0) & (df[negative_col] == 0)).astype(int)
    df["charge_state_zwitterionic"] = ((df[positive_col] == 1) & (df[negative_col] == 1)).astype(int)

    df = df.drop(columns=[positive_col, negative_col])

    engineered_columns = [
        "has_aromatic_ring",
        "has_two_aromatics",
        "charge_state_anion",
        "charge_state_cation",
        "charge_state_neutral",
        "charge_state_zwitterionic",
    ]

    base_features = df.drop(
        columns=[target, "number of aromatic rings", *engineered_columns],
        errors="ignore",
    )

    X = pd.concat([base_features, df[engineered_columns]], axis=1)
    X.columns = make_unique_columns(X.columns)
    y = df[target].astype(float)

    return X, y, df


def complete_case_filter(X: pd.DataFrame, y: pd.Series, target: str) -> tuple[pd.DataFrame, pd.Series]:
    merged = pd.concat([X, y.rename(target)], axis=1).replace([np.inf, -np.inf], np.nan)
    n_before = len(merged)
    merged = merged.dropna(axis=0, how="any").copy()
    n_after = len(merged)
    print(f"Dropped {n_before - n_after} rows due to missing or infinite values. Kept {n_after} rows.")

    X_clean = merged.drop(columns=[target])
    y_clean = merged[target].astype(float)
    return X_clean, y_clean


def split_and_scale(X: pd.DataFrame, y: pd.Series, cfg: Config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    return X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled


def fit_random_forest(X_train_scaled: pd.DataFrame, y_train: pd.Series, cfg: Config) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=cfg.rf_trees,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model


def report_test_performance(y_test: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test.to_numpy() - y_pred)))
    print(f"Test R² = {r2:.4f}")
    print(f"Test RMSE = {rmse:.4f}")
    print(f"Test MAE = {mae:.4f}")
    return r2, rmse, mae


def run_y_randomization(X: pd.DataFrame, y: pd.Series, cfg: Config) -> None:
    print("\n--- Y-randomization test ---")
    rng = np.random.default_rng(cfg.seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=cfg.seed, n_jobs=-1)),
    ])
    kf = KFold(n_splits=cfg.cv_folds_y_randomization, shuffle=True, random_state=cfg.seed)

    shuffled_scores = []
    for _ in range(cfg.y_randomization_runs):
        y_shuffled = rng.permutation(y.to_numpy())
        score = cross_val_score(pipeline, X, y_shuffled, cv=kf, scoring="r2", n_jobs=-1).mean()
        shuffled_scores.append(score)

    print("Shuffled R² scores:", np.round(shuffled_scores, 3))
    print("Mean shuffled R²:", float(np.mean(shuffled_scores)))


def run_cross_validation_check(X: pd.DataFrame, y: pd.Series, cfg: Config, out_dir: Path) -> None:
    print("\n--- 10-fold cross-validation check (RF) ---")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=cfg.seed, n_jobs=-1)),
    ])
    kf = KFold(n_splits=cfg.cv_folds_model_check, shuffle=True, random_state=cfg.seed)

    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))

    cv_df = pd.DataFrame({"R2": r2_scores, "RMSE": rmse_scores})
    cv_df.to_csv(out_dir / "rf_10fold_cv_scores.csv", index=False)
    print("Median CV R²:", float(np.median(r2_scores)))
    print("Median CV RMSE:", float(np.median(rmse_scores)))


def plot_applicability_domain(X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame, out_dir: Path) -> np.ndarray:
    print("\n--- Applicability domain ---")
    covariance_model = EmpiricalCovariance().fit(X_train_scaled)
    mahalanobis_distances = covariance_model.mahalanobis(X_test_scaled)

    plt.figure(figsize=(6, 4))
    plt.hist(mahalanobis_distances, bins=30, edgecolor="black")
    plt.title("Mahalanobis distance of test samples")
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Frequency")
    save_current_figure(out_dir / "applicability_domain_hist.png")

    return mahalanobis_distances


def plot_prediction_uncertainty(model: RandomForestRegressor, X_test_scaled: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, out_dir: Path) -> np.ndarray:
    print("\n--- Prediction uncertainty from tree spread ---")
    X_test_array = X_test_scaled.to_numpy()
    tree_predictions = np.stack([tree.predict(X_test_array) for tree in model.estimators_])
    std_devs = np.std(tree_predictions, axis=0)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, c=std_devs, cmap="viridis", edgecolor="k")
    plt.colorbar(label="Prediction standard deviation")
    plt.xlabel("Actual log Kd")
    plt.ylabel("Predicted log Kd")
    plt.title("Prediction uncertainty (RF)")
    save_current_figure(out_dir / "rf_prediction_uncertainty.png")

    return std_devs


def build_residual_table(df_processed: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, out_dir: Path) -> pd.DataFrame:
    print("\n--- Residual diagnostics ---")
    residuals = y_test.to_numpy() - y_pred
    outlier_threshold = 2 * np.std(residuals)
    is_outlier = np.abs(residuals) > outlier_threshold
    print(f"Found {int(is_outlier.sum())} outliers (|residual| > {outlier_threshold:.3f}).")

    metadata_columns = [
        "number of aromatic rings",
        "has_aromatic_ring",
        "has_two_aromatics",
        "charge_state_anion",
        "charge_state_cation",
        "charge_state_neutral",
        "charge_state_zwitterionic",
    ]
    available_metadata = [col for col in metadata_columns if col in df_processed.columns]
    missing_metadata = sorted(set(metadata_columns) - set(available_metadata))
    if missing_metadata:
        warnings.warn(f"Missing metadata columns for residual table: {missing_metadata}")

    residual_df = pd.DataFrame(index=y_test.index)
    residual_df["Actual"] = y_test.values
    residual_df["Predicted"] = y_pred
    residual_df["Residual"] = residuals
    residual_df["AbsError"] = np.abs(residuals)
    residual_df = residual_df.join(df_processed[available_metadata], how="left")

    def assign_charge_class(row: pd.Series) -> str:
        if row.get("charge_state_anion", 0) == 1:
            return "Anionic"
        if row.get("charge_state_cation", 0) == 1:
            return "Cationic"
        if row.get("charge_state_neutral", 0) == 1:
            return "Neutral"
        if row.get("charge_state_zwitterionic", 0) == 1:
            return "Zwitterionic"
        return "Unlabeled"

    residual_df["ChargeClass"] = residual_df.apply(assign_charge_class, axis=1)
    if "number of aromatic rings" in residual_df.columns:
        residual_df["Aromaticity"] = np.where(
            residual_df["number of aromatic rings"] >= 1,
            "Aromatic ring",
            "No ring",
        )

    residual_df.to_csv(out_dir / "test_residuals_with_meta.csv", index=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(residual_df["Predicted"], residual_df["Residual"], edgecolor="k")
    plt.axhline(0, linestyle="--", linewidth=1, color="k")
    plt.xlabel("Predicted log Kd")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title("Residual plot")
    save_current_figure(out_dir / "residual_plot_scatter.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=residual_df, x="ChargeClass", y="Residual")
    plt.axhline(0, linestyle="--", linewidth=1, color="k")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title("Residuals by charge class")
    save_current_figure(out_dir / "residuals_box_by_charge.png")

    if "Aromaticity" in residual_df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=residual_df, x="Aromaticity", y="Residual")
        plt.axhline(0, linestyle="--", linewidth=1, color="k")
        plt.ylabel("Residual (Actual − Predicted)")
        plt.title("Residuals by aromaticity")
        save_current_figure(out_dir / "residuals_box_by_aromaticity.png")

    return residual_df


def run_shap_analysis(model: RandomForestRegressor, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame, out_dir: Path):
    print("\n--- SHAP analysis ---")
    if not SHAP_AVAILABLE:
        print("SHAP is not available in this environment. Skipping SHAP analysis.")
        return None

    try:
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled, check_additivity=False)

        focus_feature = "Log S (mol/L) at pH 7"
        if focus_feature in X_test_scaled.columns:
            shap.plots.scatter(shap_values[:, focus_feature], color=shap_values, show=False)
            plt.title(f"SHAP dependence: {focus_feature}")
            save_current_figure(out_dir / "shap_logS_dependence.png")

        shap.summary_plot(shap_values, X_test_scaled, show=False, max_display=20)
        save_current_figure(out_dir / "shap_summary_plot.png")
        return shap_values

    except Exception as exc:
        print(f"SHAP failed: {exc!r}")
        return None


def plot_partial_dependence(model: RandomForestRegressor, X_test_scaled: pd.DataFrame, out_dir: Path) -> None:
    print("\n--- Partial dependence plots ---")
    selected_features = [
        "Log S (mol/L) at pH 7",
        "molecular weight (g/mol)",
        "O wt%",
        "surface area (m2/g)",
        "delta_PZCpH",
    ]
    selected_indices = [i for i, name in enumerate(X_test_scaled.columns) if name in selected_features]

    if selected_indices:
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(
            model,
            X_test_scaled,
            features=selected_indices,
            feature_names=list(X_test_scaled.columns),
            ax=ax,
        )
        save_current_figure(out_dir / "partial_dependence_selected.png")
    else:
        print("Selected PDP features were not found. Skipping selected PDP figure.")

    fig, ax = plt.subplots(figsize=(16, 12))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test_scaled,
        features=list(range(X_test_scaled.shape[1])),
        feature_names=list(X_test_scaled.columns),
        ax=ax,
    )
    save_current_figure(out_dir / "partial_dependence_all_features.png")


def compute_distance_correlation(X_train_scaled: pd.DataFrame, y_train: pd.Series, out_dir: Path) -> pd.DataFrame:
    print("\n--- Distance correlation with log Kd (training set, dcor) ---")
    results = []
    y_array = y_train.to_numpy()

    for column in X_train_scaled.columns:
        score = float(dcor.distance_correlation(X_train_scaled[column].to_numpy(), y_array))
        results.append((column, score))

    dcor_df = pd.DataFrame(results, columns=["Feature", "Distance Correlation"])
    dcor_df = dcor_df.sort_values("Distance Correlation", ascending=False).reset_index(drop=True)
    dcor_df.to_excel(out_dir / "distance_correlation_results.xlsx", index=False)

    plt.figure(figsize=(10, max(6, 0.35 * len(dcor_df))))
    plt.barh(dcor_df["Feature"], dcor_df["Distance Correlation"])
    plt.gca().invert_yaxis()
    plt.xlabel("Distance correlation")
    plt.title("Distance correlation with log Kd (training set)")
    save_current_figure(out_dir / "distance_correlation_plot.png")

    return dcor_df


def compute_mutual_information(X_train_scaled: pd.DataFrame, y_train: pd.Series, seed: int, out_dir: Path) -> pd.DataFrame:
    print("\n--- Mutual information with log Kd (training set) ---")
    mi_scores = mutual_info_regression(X_train_scaled.values, y_train.values, random_state=seed)
    mi_df = pd.DataFrame({
        "Feature": X_train_scaled.columns,
        "Mutual Information": mi_scores,
    }).sort_values("Mutual Information", ascending=False).reset_index(drop=True)
    mi_df.to_excel(out_dir / "mutual_information_scores.xlsx", index=False)

    plt.figure(figsize=(10, max(6, 0.35 * len(mi_df))))
    plt.barh(mi_df["Feature"], mi_df["Mutual Information"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mutual information")
    plt.title("Mutual information with log Kd (training set)")
    save_current_figure(out_dir / "mutual_information_plot.png")

    return mi_df


def run_clustering(X_all: pd.DataFrame, scaler: StandardScaler, df_processed: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    print("\n--- KMeans clustering ---")
    X_all_scaled = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=X_all.columns)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=cfg.seed)
    clusters = kmeans.fit_predict(X_all_scaled)

    pca = PCA(n_components=2, random_state=cfg.seed)
    X_pca = pca.fit_transform(X_all_scaled.values)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set1", edgecolor="k")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title("KMeans clusters (PCA view)")
    save_current_figure(out_dir / "clustering_kmeans_pca.png")

    x_name = "log D (L/kg)"
    y_name = "estimated log KOC  (L/kg)"
    if x_name in df_processed.columns and y_name in df_processed.columns:
        aligned_xy = df_processed.loc[X_all.index, [x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        cluster_series = pd.Series(clusters, index=X_all.index)
        aligned_clusters = cluster_series.loc[aligned_xy.index].to_numpy()

        plt.figure(figsize=(6, 5))
        plt.scatter(aligned_xy[x_name], aligned_xy[y_name], c=aligned_clusters, cmap="Set1", edgecolor="k")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("KMeans clusters on original axes")
        save_current_figure(out_dir / "clustering_kmeans_original_axes.png")


def compute_permutation_importance(model: RandomForestRegressor, X_test_scaled: pd.DataFrame, y_test: pd.Series, cfg: Config, out_dir: Path) -> pd.DataFrame:
    print("\n--- Permutation importance ---")
    result = permutation_importance(
        model,
        X_test_scaled,
        y_test,
        n_repeats=30,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "Feature": X_test_scaled.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std,
    }).sort_values("Importance Mean", ascending=True).reset_index(drop=True)
    perm_df.to_csv(out_dir / "permutation_importance_rf.csv", index=False)

    plt.figure(figsize=(10, max(6, 0.35 * len(perm_df))))
    plt.barh(perm_df["Feature"], perm_df["Importance Mean"], xerr=perm_df["Importance Std"])
    plt.xlabel("Permutation importance (mean)")
    plt.title("Permutation feature importance (RF)")
    save_current_figure(out_dir / "permutation_importance_rf.png")

    return perm_df


def plot_learning_curve(X: pd.DataFrame, y: pd.Series, cfg: Config, out_dir: Path) -> None:
    print("\n--- Learning curve ---")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=cfg.rf_trees, random_state=cfg.seed, n_jobs=-1)),
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cfg.cv_folds_learning_curve,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
    plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV score")
    plt.xlabel("Training set size")
    plt.ylabel("R² score")
    plt.title("Learning curve (RF, no imputation)")
    plt.legend()
    save_current_figure(out_dir / "learning_curve_rf.png")


def save_model_objects(model: RandomForestRegressor, scaler: StandardScaler, out_dir: Path) -> None:
    joblib.dump(model, out_dir / "rf_model_seed42_no_impute.pkl")
    joblib.dump(scaler, out_dir / "scaler_seed42_no_impute.pkl")
    print("Saved model and scaler objects.")


def compute_adsorbent_influence(
    shap_values,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    df_processed: pd.DataFrame,
    out_dir: Path,
) -> None:
    print("\n--- SHAP-based adsorbent influence ---")
    if shap_values is None:
        print("SHAP values are not available. Skipping adsorbent influence analysis.")
        return

    adsorbent_features = ["surface area (m2/g)", "O wt%", "delta_PZCpH"]
    if "surface area (m2/g)" not in X_test_scaled.columns and "surface area (m^2/g)" in X_test_scaled.columns:
        adsorbent_features[0] = "surface area (m^2/g)"

    missing_features = [feature for feature in adsorbent_features if feature not in X_test_scaled.columns]
    if missing_features:
        print("Missing adsorbent features in model inputs:", missing_features)

    shap_matrix = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    abs_shap = np.abs(shap_matrix)

    per_feature_abs_shap: dict[str, np.ndarray] = {}
    for feature in adsorbent_features:
        if feature in X_test_scaled.columns:
            feature_position = list(X_test_scaled.columns).index(feature)
            per_feature_abs_shap[feature] = abs_shap[:, feature_position]

    if not per_feature_abs_shap:
        print("No adsorbent features were found in the model input columns. Skipping.")
        return

    adsorbent_sum = np.sum([per_feature_abs_shap[name] for name in per_feature_abs_shap], axis=0)
    total_abs_shap = abs_shap.sum(axis=1)
    influence_fraction = np.divide(
        adsorbent_sum,
        total_abs_shap,
        out=np.zeros_like(adsorbent_sum),
        where=total_abs_shap > 0,
    )

    influence_df = pd.DataFrame(index=y_test.index)
    influence_df["Actual"] = y_test.values
    influence_df["Predicted"] = y_pred
    influence_df["Residual"] = y_test.values - y_pred
    for feature_name, values in per_feature_abs_shap.items():
        influence_df[f"ABS_SHAP__{feature_name}"] = values
    influence_df["ABS_SHAP__AdsorbentSum"] = adsorbent_sum
    influence_df["ABS_SHAP__Total"] = total_abs_shap
    influence_df["AdsorbentInfluence_Fraction"] = influence_fraction

    def label_charge(row: pd.Series) -> str:
        if row.get("charge_state_anion", 0) == 1:
            return "anionic"
        if row.get("charge_state_cation", 0) == 1:
            return "cationic"
        if row.get("charge_state_neutral", 0) == 1:
            return "neutral"
        if row.get("charge_state_zwitterionic", 0) == 1:
            return "zwitterionic"
        return "unlabeled"

    if all(
        col in df_processed.columns
        for col in [
            "charge_state_anion",
            "charge_state_cation",
            "charge_state_neutral",
            "charge_state_zwitterionic",
        ]
    ):
        df_processed = df_processed.copy()
        df_processed["charge_class"] = df_processed.apply(label_charge, axis=1)

    if "number of aromatic rings" in df_processed.columns:
        df_processed = df_processed.copy()
        df_processed["aromaticity"] = np.where(df_processed["number of aromatic rings"] >= 1, "aromatic", "nonaromatic")

    category_columns = [col for col in ["charge_class", "aromaticity"] if col in df_processed.columns]
    if category_columns:
        influence_df = influence_df.join(df_processed[category_columns], how="left")

    if {"charge_class", "aromaticity"}.issubset(influence_df.columns):
        influence_df["charge_x_aromatic"] = influence_df["charge_class"] + " " + influence_df["aromaticity"]

    influence_df.to_csv(out_dir / "adsorbent_influence_by_compound.csv", index=True)

    top_n = min(15, len(influence_df))
    top_df = influence_df.sort_values("ABS_SHAP__AdsorbentSum", ascending=False).head(top_n)
    plt.figure(figsize=(9, 6))
    plt.barh(top_df.index.astype(str), top_df["ABS_SHAP__AdsorbentSum"])
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute SHAP (surface area + O wt% + ΔPZCpH)")
    plt.title("Top compounds most influenced by adsorbent properties")
    save_current_figure(out_dir / "top_compounds_adsorbent_influence.png")

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        influence_df["Actual"],
        influence_df["Predicted"],
        c=influence_df["AdsorbentInfluence_Fraction"],
        cmap="viridis",
        edgecolor="k",
    )
    plt.colorbar(scatter, label="Fraction of total |SHAP| from adsorbent features")
    lower = float(influence_df["Actual"].min())
    upper = float(influence_df["Actual"].max())
    plt.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    plt.xlabel("Actual log Kd (L/kg)")
    plt.ylabel("Predicted log Kd (L/kg)")
    plt.title("Per-compound adsorbent influence")
    save_current_figure(out_dir / "actual_vs_pred_colored_by_adsorbent_influence.png")

    if "charge_x_aromatic" in influence_df.columns:
        ordered_groups = (
            influence_df.groupby("charge_x_aromatic")["ABS_SHAP__AdsorbentSum"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        plt.figure(figsize=(10, 5))
        sns.violinplot(
            data=influence_df,
            x="charge_x_aromatic",
            y="ABS_SHAP__AdsorbentSum",
            order=ordered_groups,
            inner="box",
            cut=0,
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Summed absolute SHAP of adsorbent features")
        plt.xlabel("Charge × aromaticity")
        plt.title("Influence of adsorbent properties by charge × aromaticity")
        save_current_figure(out_dir / "shap_adsorbent_by_crosscategory_violin.png")

        component_columns = [col for col in influence_df.columns if col.startswith("ABS_SHAP__") and col not in {"ABS_SHAP__AdsorbentSum", "ABS_SHAP__Total"}]
        stacked_means = influence_df.groupby("charge_x_aromatic")[component_columns].mean().loc[ordered_groups]
        plt.figure(figsize=(10, 5))
        running_bottom = np.zeros(len(stacked_means))
        for column in stacked_means.columns:
            plt.bar(
                stacked_means.index,
                stacked_means[column].values,
                bottom=running_bottom,
                label=column.replace("ABS_SHAP__", ""),
            )
            running_bottom += stacked_means[column].values
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean absolute SHAP")
        plt.xlabel("Charge × aromaticity")
        plt.title("Decomposition of adsorbent influence by charge × aromaticity")
        plt.legend(title="Feature")
        save_current_figure(out_dir / "shap_adsorbent_crosscategory_stacked.png")


def main() -> None:
    cfg = Config()
    out_dir = Path(__file__).resolve().parent / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(cfg)
    X_raw, y_raw, df_processed = build_features(df_raw, cfg.target)
    X, y = complete_case_filter(X_raw, y_raw, cfg.target)

    X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled = split_and_scale(X, y, cfg)

    model = fit_random_forest(X_train_scaled, y_train, cfg)
    y_pred = model.predict(X_test_scaled)
    report_test_performance(y_test, y_pred)

    run_y_randomization(X, y, cfg)
    run_cross_validation_check(X, y, cfg, out_dir)

    plot_applicability_domain(X_train_scaled, X_test_scaled, out_dir)
    plot_prediction_uncertainty(model, X_test_scaled, y_test, y_pred, out_dir)
    build_residual_table(df_processed, y_test, y_pred, out_dir)

    shap_values = run_shap_analysis(model, X_train_scaled, X_test_scaled, out_dir)
    plot_partial_dependence(model, X_test_scaled, out_dir)

    compute_distance_correlation(X_train_scaled, y_train, out_dir)
    compute_mutual_information(X_train_scaled, y_train, cfg.seed, out_dir)

    run_clustering(X, scaler, df_processed, cfg, out_dir)
    compute_permutation_importance(model, X_test_scaled, y_test, cfg, out_dir)
    plot_learning_curve(X, y, cfg, out_dir)
    save_model_objects(model, scaler, out_dir)
    compute_adsorbent_influence(shap_values, X_test_scaled, y_test, y_pred, df_processed, out_dir)

    print("\nFinished successfully.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
