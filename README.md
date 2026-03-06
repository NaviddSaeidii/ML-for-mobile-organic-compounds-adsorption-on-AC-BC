# ML for mobile organic compounds adsorption on AC/BC

**Manuscript title:** Modelling adsorption of mobile organic compounds on activated carbon and biochar using machine learning

## Abstract
We present an interpretable machine-learning model to predict adsorption of mobile organic compounds (log KOC (soil organic carbon/water partition coefficient) < 4) on activated carbon (AC) and biochar (BC) at environmentally relevant, low concentrations. The model uses molecular descriptors (molecular weight, charge state at pH 7, aromatic rings, log S and log Dow at pH 7, McGowan molar volume, log KOC) augmented with engineered features (binary charge-state indicators; aromatic ring flag) and key adsorbent properties (specific surface area, O wt%, and delta_PZCpH (PZC − experimental pH). Trained on 509 literature log Kd values (Ce < 5 µg/L) for 75 compounds, models using molecular properties alone underperform, while adding adsorbent descriptors yields accurate predictions (best: Random Forest, R² ≈ 0.82, RMSE ≈ 0.33 for log (Kd/[L/kg]) range 2.3 to 7.4. Interpretation (distance correlation, mutual information, SHAP) identifies hydrophobicity proxies and surface area as dominant drivers, with smaller global effects from formal charge; log KOC (as a mobility/soprtion proxy) shows the largest molecular impact after adsorbent properties, notably surface area and delta_PZCpH. Non-aromatic compounds—especially anionic and neutral non-aromatics—are most influenced by adsorbent characteristics. Anionic species show the highest deviations; neutrals the lowest. Performance on unseen independent experiments aligns closely with the 1:1 line, with greater uncertainty at response extremes. This general model enables rapid screening of AC/BC systems for removing mobile organics from water.

## 🚀 Quick start
```bash
git clone https://github.com/<your-username>/ml-mobile-organics-adsorption-ac-bc.git
cd ml-mobile-organics-adsorption-ac-bc

# Option A: conda
conda env create -f environment.yml
conda activate manuscript-env

# Option B: pip/venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 📂 Structure
```
.
├── src/                 # Python modules and scripts
├── notebooks/           # Jupyter notebooks for analysis/figures
├── data/
│   ├── raw/             # Unmodified/raw inputs (not tracked in git)
│   └── processed/       # Cleaned/derived data (small files only)
├── results/             # Outputs, tables
├── figures/             # Final figures for the manuscript
├── environment.yml
├── requirements.txt
├── CITATION.cff
├── LICENSE              # Code license (MIT)
├── LICENSE-Data         # Data/doc license (CC BY 4.0)
└── README.md
```

## 🧪 Reproducibility
- Pin exact dependencies: `pip freeze > requirements-lock.txt` or `conda env export > environment-lock.yml`.
- Use **Git LFS** or an external host for large files (>100 MB); keep only small samples in `data/`.
- Scripts that regenerate figures should write to `figures/` with informative names (e.g., `fig2_sorption_isotherm.png`).

## 📣 Citation
A `CITATION.cff` file is provided. After creating a release, consider linking Zenodo to mint a DOI.

## 🔒 Licensing
- **Code:** MIT License (permissive, widely used).
- **Data & docs:** CC BY 4.0 (credit required).
If you prefer a single license for everything, keep MIT only and delete `LICENSE-Data`.

## 🔗 Repository URL
Replace with your real path after creation:
```
https://github.com/<your-username>/ml-mobile-organics-adsorption-ac-bc
```
