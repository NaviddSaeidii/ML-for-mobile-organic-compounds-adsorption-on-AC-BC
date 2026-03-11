# ML for mobile organic compounds adsorption on AC/BC

This repository contains the files required to reproduce the machine-learning workflow associated with the manuscript on adsorption of mobile organic compounds on activated carbon (AC) and biochar (BC).

The public GitHub repository includes only the following files:

1. `run_single_csv.py` together with `merged_train_external.csv`
2. `rf_predict_logKd_excel.py` together with `Prediction_Input_Template.xlsx` and `merged_train_external.csv`
3. `SI_Excel_compounds and adsorption info.xlsx`

## Repository contents

- `run_single_csv.py`  
  Script for training and evaluating machine-learning models using the provided CSV dataset.

- `merged_train_external.csv`  
  Dataset used with `run_single_csv.py` and as the training dataset for `rf_predict_logKd_excel.py`.

- `rf_predict_logKd_excel.py`  
  Script for predicting logKd values from Excel input data.

- `Prediction_Input_Template.xlsx`  
  Excel template for user input when running `rf_predict_logKd_excel.py`.

- `SI_Excel_compounds and adsorption info.xlsx`  
  Supplementary information file containing compound and adsorption-related data.

- `environment.yml`  
  Conda environment file listing the required Python packages.

- `CITATION.cff`  
  Citation metadata for this repository.

- `LICENSE`  
  License file for repository reuse.

## Requirements

The scripts were prepared for Python 3 and require common scientific Python packages including:

- pandas
- numpy
- scikit-learn
- openpyxl
- joblib
- matplotlib
- seaborn

Optional:
- shap

A conda environment can be created from:

```bash
conda env create -f environment.yml
conda activate ml-adsorption
Usage
1. Run model workflow with CSV input
python run_single_csv.py

This script uses:

merged_train_external.csv

Please keep the script and CSV file in the same folder, or adapt the file path in the script.

2. Predict logKd from Excel input
python rf_predict_logKd_excel.py --train

This script uses:

Prediction_Input_Template.xlsx as input template

merged_train_external.csv as training dataset

The input Excel file can be filled for one row or multiple rows. The first row is provided as an example.

3. Supplementary information

The file SI_Excel_compounds and adsorption info.xlsx is included as supplementary data associated with the manuscript.

Notes

Only the files listed above are included in this GitHub version.

Other working documents and internal files were intentionally excluded from the public repository.

File paths may need to be adapted depending on the local execution environment.

Citation

Please cite the associated manuscript if you use these files.

Additional repository citation metadata is provided in CITATION.cff.
