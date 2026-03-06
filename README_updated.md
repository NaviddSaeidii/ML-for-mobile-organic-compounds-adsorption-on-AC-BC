# Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning

This repository contains the code and data supporting the manuscript:

Navid Saeidi, David J. Vicente, Sampriti Chaudhuri, and Anett Georgi, “Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning”.

## Overview

This study develops an interpretable machine-learning framework to predict adsorption coefficients (log Kd) of mobile and very mobile organic compounds on activated carbon (AC) and biochar (BC) at environmentally relevant low concentrations. The dataset used for model development was compiled from the literature and contains 509 log Kd values for 75 compounds under comparable conditions. In addition, an independent experimental dataset containing 23 log Kd values for 14 compounds is included for external evaluation and prediction-related workflows. The framework combines molecular descriptors, engineered descriptors for charge state and aromaticity, and key adsorbent descriptors including specific surface area, O wt%, and delta_PZCpH. Random Forest showed the best overall predictive performance in the manuscript. fileciteturn4file1turn4file6turn4file13turn4file17

## Repository contents

- `train_compare_models.py`  
  Trains and compares Random Forest, Ridge, and SVR models using the main processed dataset and saves model-comparison figures and metrics.

- `RF_full_analysis.py`  
  Performs detailed Random Forest analysis, including train/test evaluation, cross-validation, SHAP analysis, dependence analysis, partial dependence plots, and diagnostic outputs.

- `rf_predict_logKd_excel_inplace.py`  
  Uses the trained Random Forest model to predict log Kd values for new inputs provided in Excel format. The script can also train the model artifacts if they are not already available.

- `Prediction_Input_Template.xlsx`  
  Template Excel file for prediction of new samples. Users should fill in the required input columns and then run the prediction script.

- `cleaned_with_deltaPZCpH_no planar.xlsx`  
  Main processed literature-derived dataset used for model development.

- `SI_Excel_compounds and adsorption info.xlsx`  
  Supplementary Excel file containing compound and adsorption information, including the independent evaluation dataset.

- `CSV SI.csv`  
  Supplementary tabular data file corresponding to the manuscript supplementary information.

- `environment.yml`  
  Conda environment file for reproducing the software environment.

- `CITATION.cff`  
  Citation metadata for this repository.

## Software environment

The manuscript reports that the study was implemented in Python 3.12 using open-source libraries including `pandas`, `seaborn`, `scikit-learn`, `shap`, `dcor`, `numpy`, and `matplotlib`. The environment file below is intended to reproduce that setup as closely as possible. fileciteturn4file17

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate adsorption-ml
```

## How to run

### 1. Compare the three models

```bash
python train_compare_models.py
```

This script reads `cleaned_with_deltaPZCpH_no planar.xlsx` and saves:
- one CSV file with model metrics
- one predicted-vs-actual figure for each model

### 2. Run the full Random Forest analysis

```bash
python RF_full_analysis.py
```

This script reads `cleaned_with_deltaPZCpH_no planar.xlsx` and writes detailed analysis outputs to the `outputs` folder.

### 3. Predict log Kd for new input rows

First, fill `Prediction_Input_Template.xlsx` with the required input values. The model expects the same numeric feature columns used during training. According to the manuscript, the required inputs are molecular descriptors, engineered charge-state and aromaticity descriptors, and adsorbent descriptors such as specific surface area, O wt%, and delta_PZCpH. fileciteturn4file10turn4file16

Then run:

```bash
python rf_predict_logKd_excel_inplace.py --train
```

This will:
- train the Random Forest model from `cleaned_with_deltaPZCpH_no planar.xlsx`
- save model artifacts as `.joblib` files
- create a backup copy of `Prediction_Input_Template.xlsx`
- write predictions into the same Excel file under the column `pred_log Kd (L/kg)`

After the first run, prediction can also be done without forced retraining:

```bash
python rf_predict_logKd_excel_inplace.py
```

Optional custom paths:

```bash
python rf_predict_logKd_excel_inplace.py --train --training "cleaned_with_deltaPZCpH_no planar.xlsx" --input "Prediction_Input_Template.xlsx"
```

## Notes on reproducibility

- The literature-based dataset is used for model development and internal train/test evaluation.
- The independent dataset in `SI_Excel_compounds and adsorption info.xlsx` corresponds to the external evaluation described in the manuscript.
- Blank cells are treated as missing values, and the modeling workflow uses complete-case analysis with no imputation. fileciteturn4file10turn4file17
- The model is intended for mobile and very mobile organic compounds similar to those represented in the training data. Application to compounds with substantially higher log KOC should be considered extrapolation. fileciteturn4file10turn4file17
- Minor differences in figure appearance, colors, or symbols may occur across systems and can be adjusted separately.

## Data and licensing

Code license: see `LICENSE`  
Data and documentation license: see `LICENSE-Data`

## Citation

Please cite the manuscript and this repository if you use the code or data.
