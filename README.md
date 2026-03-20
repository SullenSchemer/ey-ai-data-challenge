# EY AI & Data Challenge 2026 — Water Quality Prediction
 
Predicting water quality in South African rivers using Landsat satellite imagery and TerraClimate climate data.
 
**Final score: 0.313 R²** | Baseline: 0.20 | Certificate threshold: 0.40 | Leaderboard top: 0.8529
 
---
 
## The Problem
 
Predict three water quality parameters at river monitoring sites:
 
| Parameter | Unit | Healthy Range |
|---|---|---|
| Total Alkalinity | mg/L | 20–200 |
| Electrical Conductance | µS/cm | < 800 |
| Dissolved Reactive Phosphorus | µg/L | < 100 |
 
**The core challenge:** Training and validation locations are in completely different geographic regions. Models that learn location-specific patterns fail. Models that learn general climate–water relationships generalize.
 
---
 
## Approach
 
**Winning feature set — Climate + Normalized Indices:**
 
| Feature | Source | Why it works |
|---|---|---|
| PET (evapotranspiration) | TerraClimate | Captures heat/aridity |
| Precipitation | TerraClimate | Dilution and runoff effects |
| Soil moisture | TerraClimate | Groundwater quality proxy |
| Water deficit, max temp, runoff | TerraClimate | Regional climate signal |
| NDMI, MNDWI | Landsat | Normalized ratios — location-independent |
| PET/precipitation ratio | Derived | Aridity index |
| Month (sin/cos encoded) | Date | Seasonal wet/dry cycles |
 
**What didn't work:** Raw Landsat band values (NIR, green, SWIR). These are absolute brightness values tied to specific locations and don't transfer to new regions.
 
**Model:** Random Forest Regressor with conservative depth settings (`max_depth=8`, `min_samples_leaf=15`) to reduce overfitting.
 
---
 
## Score History
 
| Version | Date | Leaderboard R² | Key Change |
|---|---|---|---|
| v1 | Mar 11 | 0.080 | 17 features, XGBoost — severe overfitting |
| v2 | Mar 11 | 0.222 | 10 features, RF vs XGBoost model competition |
| v3 | Mar 11 | 0.228 | Added TerraClimate precipitation |
| v4 | Mar 12 | **0.289** | Strategy comparison — dropped raw Landsat bands, indices + climate only, 70/30 split |
| v5 | Mar 12 | 0.219 | Log-transformed targets + trained on all data (no split) — hurt generalization |
| v6 | Mar 12 | 0.281 | Added soil moisture from TerraClimate |
| v7 | Mar 12 | 0.119 | Strategy B with soil + water deficit (def) — overfit |
| v8 | Mar 13 | **0.313** | Added tmax (max temperature) |
| v9 | Mar 13 | 0.304 | Added runoff (q) — marginal negative effect |
 
**Leaderboard context:**
 
| | R² |
|---|---|
| Challenge baseline | 0.200 |
| My best (v8) | **0.313** |
| Certificate threshold | 0.400 |
| Top 10 range | 0.765 – 0.853 |
 
Scores above 0.4 required more advanced approaches — likely spatial feature engineering, gradient boosting with tuned hyperparameters, or ensemble methods. The top score (0.853) suggests the geographic distribution shift is solvable with the right feature set.
 
**Key learnings:**
- Internal CV R² (~0.49) consistently overestimated leaderboard performance (~0.31) — geographic distribution shift between train and validation regions is the core problem
- v4 → v5 regression: log-transforming targets and removing the train/test split improved internal scores but hurt leaderboard performance
- TerraClimate variables helped incrementally (v3, v6, v8) but with diminishing returns — the ceiling for this feature set was ~0.31
- v7's collapse (0.119) shows that feature combinations can interact badly; adding more features is not always better
 
---
 
## Data Sources
 
| Dataset | Source | Variables Used |
|---|---|---|
| Water quality labels | UNEP via EY | Total Alkalinity, EC, DRP |
| Landsat 8/9 | NASA via Microsoft Planetary Computer | NDMI, MNDWI |
| TerraClimate | Microsoft Planetary Computer | PET, ppt, soil, def, tmax, q |
 
> **Note:** EY-provided datasets (water quality labels, pre-extracted Landsat and TerraClimate features, submission template) are excluded from this repository per the [competition Terms & Conditions](https://challenge.ey.com). To reproduce results, register for the challenge and download from the [EY Challenge portal](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/data-description).
 
---
 
## Project Structure
 
```
├── data/
│   ├── terraclimate_training_multi.csv     # Self-extracted climate variables (ppt, soil, def, tmax, q)
│   ├── terraclimate_validation_multi.csv   # Self-extracted climate variables for validation
│   ├── water_quality_training_dataset.csv  # ⛔ Not included (EY confidential)
│   ├── landsat_features_training.csv       # ⛔ Not included (EY confidential)
│   ├── landsat_features_validation.csv     # ⛔ Not included (EY confidential)
│   ├── terraclimate_features_training.csv  # ⛔ Not included (EY confidential)
│   └── terraclimate_features_validation.csv# ⛔ Not included (EY confidential)
├── submissions/
│   ├── submission_final.csv                # Best predictions (v8 — 0.313 R²)
│   └── submission_template.csv             # ⛔ Not included (EY confidential)
├── .gitignore
├── extract_terraclimate.py                 # TerraClimate API extraction script
├── ey_water_quality_model.py               # Final prediction model
├── README.md
└── requirements.txt
```
 
---
 
## Usage
 
```bash
# Install dependencies
pip install -r requirements.txt
 
# Run the prediction model (requires EY datasets in data/ folder)
python ey_water_quality_model.py
 
# (Optional) Re-extract TerraClimate data from public API
python extract_terraclimate.py
```
 
---
 
## Tech Stack
 
Python 3.11 · Pandas · NumPy · Scikit-learn · Microsoft Planetary Computer API
 
---
 
*EY AI & Data Challenge 2026 — submitted March 2026*
