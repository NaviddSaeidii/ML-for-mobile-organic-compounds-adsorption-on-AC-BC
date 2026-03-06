# Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning

This repository contains the code and data supporting the manuscript:

Saeidi, Navid, et al. “Modeling adsorption of mobile organic compounds on activated carbon and biochar using machine learning”.

## Overview

This study develops interpretable machine-learning models to predict adsorption coefficients (log Kd) of mobile organic compounds on activated carbon and biochar at environmentally relevant low concentrations. The work is based on a harmonized literature dataset with 509 log Kd values for 75 compounds, together with an independent experimental dataset containing 23 log Kd values for 14 compounds for external evaluation.

The modeling framework combines molecular descriptors with adsorbent descriptors, including specific surface area, oxygen content, and delta_PZCpH. In addition, engineered binary variables were used to represent the pH-dependent charge state and aromaticity of the adsorbate.

## Repository contents

- `train_compare_models.py`  
  Trains and compares Random Forest, Ridge, and SVR models using the main processed dataset.

- `RF_full_analysis.py`  
  Performs detailed Random Forest analysis, including model evaluation, cross-validation, feature importance, SHAP analysis, partial dependence analysis, and diagnostic outputs.

- `rf_predict_logKd_excel_inplace.py`  
  Uses the trained Random Forest model to predict log Kd values for new inputs provided in an Excel file and writes the results back into the same file.

- `cleaned_with_deltaPZCpH_no planar.xlsx`  
  Main processed dataset used for model development.

- `Prediction_Input_Template.xlsx`  
  Example input file for prediction with `rf_predict_logKd_excel_inplace.py`.

- `CSV SI.csv`  
  Supplementary data file. Depending on system settings, this file may use semicolon separators.

- `SI_Excel_compounds and adsorption info.xlsx`  
  Supplementary Excel file containing compound and adsorption information, including the independent evaluation dataset.

- `environment.yml`  
  Conda environment file for reproducing the software environment.

- `CITATION.cff`  
  Citation metadata for the repository.

## Software environment

The analysis was performed in Python 3.12 using packages including:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap
- dcor
- joblib
- openpyxl

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate adsorption-ml
```

## How to run

Run model comparison:

```bash
python "train_compare_models.py"
```

Run detailed Random Forest analysis (first creat the conda environment, then run the analysis):

```bash
python "RF_full_analysis.py"
```

Run prediction on new input data:

```bash
python "rf_predict_logKd_excel_inplace.py"
```

## Prediction workflow

1. Open `Prediction_Input_Template.xlsx`
2. Enter the required input descriptor(s) for the new compounds and adsorbents. It works for one signle row of inputs (for one adsorption) or for a series of inputs (several adsorptions). 
3. Save the file
4. Run:

```bash
python "rf_predict_logKd_excel_inplace.py"
```

The script writes the predicted log Kd values back into the Excel file (in column M).

## Notes on reproducibility

- The repository contains the processed training dataset used for model development.
- The supplementary Excel file includes the independent external evaluation dataset.
- Minor differences in figure appearance, colors, or symbols may occur and can be adjusted separately.
- The prediction script is intended for applying the trained model to new input data provided in Excel format.

## Data and licensing

Code and data license: see LICENSE

## Citation

Please cite the manuscript and this repository if you use the code or data.
