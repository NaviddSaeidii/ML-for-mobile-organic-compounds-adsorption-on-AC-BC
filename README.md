# Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning

This repository contains the code and data supporting the manuscript:

Navid Saeidi, David J. Vicente, Sampriti Chaudhuri, and Anett Georgi. “Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning”.

## Overview

This repository provides an interpretable machine-learning workflow to predict adsorption coefficients, log Kd (L/kg), of mobile and very mobile organic compounds on activated carbon (AC) and biochar (BC) at environmentally relevant low concentrations. The workflow is based on a harmonized literature dataset with 509 log Kd values for 74 compounds and an independent experimental evaluation dataset with 23 log Kd values for 14 compounds.

The analysis combines:

- molecular descriptors
- engineered descriptors for charge state and aromaticity
- adsorbent descriptors: specific surface area, O wt%, and delta_PZCpH

Two feature settings are compared:

1. molecular descriptors only
2. molecular plus adsorbent descriptors

Three regression models are evaluated:

- Random Forest (RF)
- Ridge Regression
- Support Vector Regression (SVR)

The main workflow reproduces the model-comparison results, Random Forest interpretation, cross-validation, residual diagnostics, applicability-domain screening, and independent evaluation described in the manuscript.

## Main analysis script

The central script in this repository is:

- `ml_analysis_single_script_with_comparison.py`

This script performs the end-to-end analysis:

- loads the processed training dataset
- engineers charge-state and aromaticity features
- builds two feature sets (molecular only, molecular + adsorbent)
- trains and compares RF, Ridge, and SVR models
- evaluates test-set performance using R², RMSE, and MAE
- generates manuscript and supplementary figures
- computes distance correlation and mutual information rankings
- performs 10-fold cross-validation for the RF model
- generates residual and applicability-domain diagnostics
- runs SHAP analysis when the `shap` package is available
- evaluates the trained RF model on the independent external dataset
- saves figures and summary tables to an output folder

## Repository contents

### Files used for the full analysis

- `ml_analysis_single_script_with_comparison.py`  
  Main end-to-end analysis script for model comparison, interpretation, diagnostics, and independent evaluation.

- `cleaned_with_deltaPZCpH_no planar.xlsx`  
  Processed literature-derived dataset used for model development.

- `SI_Excel_compounds and adsorption info.xlsx`  
  Supplementary Excel file containing the combined train/evaluation information, including the independent evaluation dataset used by the main script.

- `CSV SI.csv`  
  Supplementary CSV file accompanying the manuscript.

- `environment.yml`  
  Conda environment definition for reproducibility.

- `CITATION.cff`  
  Citation metadata for the repository.

- `LICENSE`  
  Repository license.

### Files used for prediction on new inputs

The repository also includes a separate prediction utility for applying the trained Random Forest workflow to new compounds or new compound–adsorbent combinations:

- `rf_predict_logKd_excel_inplace.py`  
  Prediction script that reads descriptor values from an Excel file, applies the same feature engineering used in the training workflow, and writes predicted log Kd values back into the Excel file.

- `Prediction_Input_Template.xlsx`  
  Input template for prediction. The user can enter descriptor values for a single case (one row) or for multiple cases (multiple rows). The script writes the predicted `log Kd (L/kg)` values into column M of the same file and also creates a backup copy before overwriting the file.

When `rf_predict_logKd_excel_inplace.py` is run for the first time, it trains the Random Forest model from `cleaned_with_deltaPZCpH_no planar.xlsx` and saves the fitted model, scaler, and feature list as `.joblib` files in the repository folder. Later runs reuse these saved files unless retraining is requested.

## Software environment

The analysis was developed in Python 3.12 and uses open-source packages including:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap (optional, for SHAP figures)
- openpyxl
- joblib

The main script computes distance correlation directly in Python and does not require the `dcor` package.

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate adsorption-ml
```

## How to run the full analysis

Place the script in the repository root together with the Excel input files, then run:

```bash
python "ml_analysis_single_script_with_comparison.py"
```

## How to run prediction for new inputs

To predict log Kd values for new cases using the Excel template, run:

```bash
python "rf_predict_logKd_excel_inplace.py"
```

Optional arguments:

```bash
python "rf_predict_logKd_excel_inplace.py" --train
python "rf_predict_logKd_excel_inplace.py" --input "Prediction_Input_Template.xlsx"
python "rf_predict_logKd_excel_inplace.py" --training "cleaned_with_deltaPZCpH_no planar.xlsx"
```

## What the main analysis script produces

The script creates an output folder named:

```text
ml_manuscript_si_outputs
```

This folder contains manuscript-ready and supplementary outputs, including:

- model-comparison plots for molecular-only and molecular-plus-adsorbent feature sets
- individual predicted-vs-actual plots for RF, Ridge, and SVR
- distance correlation and mutual information summary tables and figures
- RF test-set plots colored by charge class and aromaticity
- RF uncertainty map
- SHAP summary plot (if `shap` is installed)
- partial dependence plots
- 10-fold CV summary
- residual diagnostics and outlier visualization
- applicability-domain histogram based on Mahalanobis distance
- independent evaluation results and figure
- CSV summaries of model performance and diagnostics

Examples of output files generated by the script include:

- `model_comparison_summary.csv`
- `distance_correlation_results.csv`
- `mutual_information_results.csv`
- `rf_10fold_cv_scores.csv`
- `test_residuals_with_meta.csv`
- `independent_evaluation_results.csv`
- `Fig_3a_model_comparison_molecular_only.png`
- `Fig_3b_model_comparison_molecular_plus_adsorbent.png`
- `Fig_4_SHAP_summary.png`
- `Fig_5_partial_dependence_selected_features.png`

## Input data expected by the main analysis script

The script expects the following files in the same directory:

- `cleaned_with_deltaPZCpH_no planar.xlsx`
- `SI_Excel_compounds and adsorption info.xlsx`

The training target is:

- `log Kd (L/kg)`

The script uses the processed dataset to construct the molecular-only and molecular-plus-adsorbent feature sets and extracts the independent evaluation subset from the supplementary Excel file.

## Notes

- Missing values are handled by complete-case filtering for the selected feature set.
- The train-test split uses `test_size=0.2` and `random_state=42`.
- The Random Forest model in the main analysis uses 200 trees.
- SHAP outputs are skipped automatically if the `shap` package is not installed.
- The workflow is designed for compounds and adsorbents represented by the processed dataset and manuscript scope.

## Reproducibility

This repository includes the processed data files and the code needed to reproduce the main comparative modeling workflow and the Random Forest interpretation pipeline described in the manuscript. Minor differences in figure appearance may occur across systems because of package versions or local rendering settings.

## Citation

Please cite the manuscript and this repository if you use the code or data.
