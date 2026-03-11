# ML for mobile organic compounds adsorption on AC/BC

This repository contains the files required to reproduce the prediction workflow associated with the manuscript on adsorption of mobile organic compounds on activated carbon (AC) and biochar (BC).

The GitHub version of this repository includes only the files intended for public sharing:

1. `run_single_csv.py` together with `merged_train_external.csv`
2. `rf_predict_logKd_excel.py` together with `Prediction_Input_Template.xlsx` and `merged_train_external.csv`
3. `SI_Excel_compounds and adsorption info.xlsx`

## Repository contents

- `run_single_csv.py`  
  Script for training and evaluating ML models and detailed analysis on the Random Forest model using the provided CSV dataset.

- `merged_train_external.csv`  
  CSV dataset used together with `run_single_csv.py`.

- `rf_predict_logKd_excel.py`  
  Script for predicting logKd values from Excel input data. In this repository version, the training dataset is set to `merged_train_external.csv`.

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

The scripts were prepared for Python 3 and require common scientific Python packages such as:

- pandas
- numpy
- scikit-learn
- openpyxl
- joblib
- matplotlib
- seaborn
- shap

A conda environment can be created from:

```bash
conda env create -f environment.yml
conda activate ml-adsorption
```

## Usage

1. Run model workflow with CSV input

Use:

`python run_single_csv.py`

This script works with:

`merged_train_external.csv`

Please keep the script and CSV file in the same folder, or adapt the file path in the script.

2. Predict logKd from Excel input

Use:

`python rf_predict_logKd_excel.py --train`

This script uses:

`Prediction_Input_Template.xlsx` as input template

`merged_train_external.csv` as training dataset

Please keep these files in the same folder as the script, or adapt the file paths if needed. The input Excel file can be filled for one set of data or for multiple-row inputs. The first row is filled as an example.

3. Supplementary information

The file `SI_Excel_compounds and adsorption info.xlsx` is included as supplementary data associated with the manuscript.

## Notes

Only the files listed above are included in this GitHub version.

File paths may need to be adapted depending on the local execution environment.

## Citation

Please cite the associated manuscript if you use these files.

In addition, repository citation metadata is provided in `CITATION.cff`.
