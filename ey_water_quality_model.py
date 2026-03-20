"""
EY AI & Data Challenge 2026 — Water Quality Prediction
=======================================================
Predicts three water quality parameters for South African rivers
using Landsat satellite imagery and TerraClimate climate data.

Best leaderboard score achieved: 0.313 (Strategy A: Climate + Indices)

Key insight: The validation set covers completely different geographic
regions than the training set. Models that rely on raw satellite band
values (which are location-specific) fail to generalize. Normalized
spectral indices (NDMI, MNDWI) and climate variables (precipitation,
soil moisture) transfer across regions and perform best.

Parameters predicted:
  - Total Alkalinity (mg/L)
  - Electrical Conductance (uS/cm)
  - Dissolved Reactive Phosphorus (ug/L)

Evaluation metric: Mean R² across all three parameters.

Usage:
  python ey_water_quality_model.py

Requirements:
  pip install pandas numpy scikit-learn
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"        # Folder containing input CSV files
OUTPUT_DIR = "submissions"  # Folder where submission CSV is saved

# The three water quality values we're trying to predict
TARGET_COLUMNS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus'
]

# Random Forest settings — kept conservative to reduce overfitting
RF_PARAMS = dict(
    n_estimators=300,   # Number of decision trees in the forest
    max_depth=8,        # How deep each tree can grow (limits memorization)
    min_samples_leaf=15, # Minimum samples needed at a leaf node (regularization)
    max_features=0.6,   # Each split considers only 60% of features (diversity)
    random_state=42,    # Reproducibility
    n_jobs=-1           # Use all CPU cores
)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("📂 Loading data...")

# Water quality measurements (ground truth labels)
water_quality = pd.read_csv(os.path.join(DATA_DIR, "water_quality_training_dataset.csv"))

# Landsat satellite features — spectral bands + indices per location/date
landsat_train = pd.read_csv(os.path.join(DATA_DIR, "landsat_features_training.csv"))
landsat_val   = pd.read_csv(os.path.join(DATA_DIR, "landsat_features_validation.csv"))

# TerraClimate features — single variable (PET) extracted by the benchmark notebook
terra_train = pd.read_csv(os.path.join(DATA_DIR, "terraclimate_features_training.csv"))
terra_val   = pd.read_csv(os.path.join(DATA_DIR, "terraclimate_features_validation.csv"))

# TerraClimate multi-variable file — additional climate variables extracted separately
# Variables: ppt (precipitation), soil (soil moisture), def (water deficit),
#            tmax (max temperature), q (runoff)
terra_train_multi = pd.read_csv(os.path.join(DATA_DIR, "terraclimate_training_multi.csv"))
terra_val_multi   = pd.read_csv(os.path.join(DATA_DIR, "terraclimate_validation_multi.csv"))

# Submission template — defines which 600 rows to predict (location + date)
submission_template = pd.read_csv(os.path.join(OUTPUT_DIR, "submission_template.csv"))

# Detect which TerraClimate variables are actually present in the multi file
# (extraction may have been partial due to API rate limits)
TERRA_VARS = [
    col for col in terra_train_multi.columns
    if col not in ['Latitude', 'Longitude', 'Sample Date']
]

print(f"   TerraClimate variables available: {TERRA_VARS}")
print(f"   Training samples: {len(water_quality):,}")
print(f"   Validation samples: {len(landsat_val):,}")
print("✅ Data loaded.\n")

# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================

def build_features(landsat_df, terra_df, terra_multi_df):
    """
    Constructs the feature matrix used for training and prediction.

    Strategy: Climate + Normalized Indices only (no raw satellite bands).
    
    Raw Landsat band values (NIR, green, SWIR) reflect the absolute 
    brightness at a specific location. These values differ between 
    training and validation regions, causing the model to learn 
    location-specific patterns rather than general water quality signals.

    Normalized indices (NDMI, MNDWI) are ratios between bands, which
    cancel out location-specific brightness differences. Climate variables
    (precipitation, soil moisture, etc.) are physically linked to water
    quality and are stable across regions.

    Parameters
    ----------
    landsat_df : DataFrame
        Landsat features (bands, indices, dates) for a set of samples.
    terra_df : DataFrame
        Single-variable TerraClimate features (PET).
    terra_multi_df : DataFrame
        Multi-variable TerraClimate features (ppt, soil, def, tmax, q).

    Returns
    -------
    DataFrame
        Feature matrix ready for model training or prediction.
    """
    features = pd.DataFrame()

    # --- Climate features ---
    # Potential Evapotranspiration: how much water could evaporate given temperature/radiation
    # High PET → hot/dry conditions → affects dissolved salts, alkalinity
    features['pet'] = terra_df['pet']

    # Add whichever additional TerraClimate variables were successfully extracted
    # e.g. ppt=precipitation, soil=soil moisture, def=water deficit, tmax, q=runoff
    for var in TERRA_VARS:
        features[var] = terra_multi_df[var]

    # --- Normalized spectral indices ---
    # NDMI (Normalized Difference Moisture Index): (NIR - SWIR16) / (NIR + SWIR16)
    # Measures vegetation and soil moisture. Already pre-computed in the Landsat file.
    features['NDMI'] = landsat_df['NDMI']

    # MNDWI (Modified Normalized Difference Water Index): (Green - SWIR16) / (Green + SWIR16)
    # Highlights open water bodies. Already pre-computed in the Landsat file.
    features['MNDWI'] = landsat_df['MNDWI']

    # --- Interaction features ---
    # PET / precipitation ratio: captures aridity
    # High ratio = dry conditions → water concentrates dissolved minerals
    if 'ppt' in features.columns:
        features['pet_ppt_ratio'] = features['pet'] / features['ppt'].replace(0, np.nan)
        # Precipitation × moisture index: wetter periods should affect water chemistry
        features['ppt_x_NDMI'] = features['ppt'] * features['NDMI']
    else:
        # Fallback if precipitation wasn't extracted
        features['pet_ppt_ratio'] = features['pet'].replace(0, np.nan)

    # --- Temporal features (cyclical encoding) ---
    # Encode month as sine/cosine so January and December are "close" to each other.
    # South African wet season: October–March. Dry season: April–September.
    # Raw month numbers (1–12) falsely imply December and January are far apart.
    sample_date = pd.to_datetime(
        landsat_df['Sample Date'], format='mixed', dayfirst=True
    )
    features['month_sin'] = np.sin(2 * np.pi * sample_date.dt.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * sample_date.dt.month / 12)

    return features


# Build feature matrices for training and validation
print("🔧 Engineering features...")
X_train_full = build_features(landsat_train, terra_train, terra_train_multi)
X_val        = build_features(landsat_val,   terra_val,   terra_val_multi)

# Fill any NaN values with the median of each column
# NaNs can occur from division (e.g. ppt=0) or missing API data
X_train_full = X_train_full.fillna(X_train_full.median(numeric_only=True))
X_val        = X_val.fillna(X_val.median(numeric_only=True))

feature_names = list(X_train_full.columns)
print(f"   Features ({len(feature_names)}): {feature_names}\n")

# ============================================================
# STEP 3 — TRAIN & EVALUATE (one model per parameter)
# ============================================================

print("🤖 Training models...\n")

cv_scores = {}  # Track cross-validation R² per parameter

for target in TARGET_COLUMNS:
    print(f"  [{target}]")

    # Extract target labels
    y = water_quality[target]

    # Hold out 30% of data for a local test score
    # This gives a rough idea of model quality, but does NOT reflect
    # leaderboard performance (validation regions are geographically different)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_full, y, test_size=0.3, random_state=42
    )

    # Standardize features: transform each column to mean=0, std=1
    # Random Forests don't strictly need scaling, but it helps Ridge/XGBoost
    # and makes cross-validation more stable
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)   # Fit on training, transform training
    X_te_s = scaler.transform(X_te)       # Transform test using same scaler

    # Train Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr_s, y_tr)

    # Local test R² (how well the model predicts the held-out 30%)
    test_r2 = r2_score(y_te, model.predict(X_te_s))

    # 5-fold cross-validation R² (more robust estimate)
    # Splits training data into 5 folds, trains on 4, tests on 1, repeats
    cv = cross_val_score(model, X_tr_s, y_tr, cv=5, scoring='r2').mean()
    cv_scores[target] = cv

    print(f"    Test R²: {test_r2:.3f} | CV R²: {cv:.3f}")

avg_cv = np.mean(list(cv_scores.values()))
print(f"\n  📊 Average CV R²: {avg_cv:.3f}")
print("  (Note: leaderboard R² will differ due to geographic distribution shift)\n")

# ============================================================
# STEP 4 — GENERATE SUBMISSION
# ============================================================
# Retrain on ALL training data (no holdout) for best possible predictions
# on the validation set.

print("🔮 Building final predictions...\n")

predictions = {}

for target in TARGET_COLUMNS:
    y = water_quality[target]

    # Scale using all available training data
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_full)
    X_va_s = scaler.transform(X_val)

    # Train on full dataset
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr_s, y)

    # Predict; clip negatives to 0 (water quality values can't be negative)
    preds = np.clip(model.predict(X_va_s), 0, None)
    predictions[target] = preds

    print(f"  ✅ {target}: mean={preds.mean():.2f}, min={preds.min():.2f}, max={preds.max():.2f}")

# ============================================================
# STEP 5 — SAVE SUBMISSION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

submission_df = pd.DataFrame({
    'Latitude':                      submission_template['Latitude'].values,
    'Longitude':                     submission_template['Longitude'].values,
    'Sample Date':                   submission_template['Sample Date'].values,
    'Total Alkalinity':              predictions['Total Alkalinity'],
    'Electrical Conductance':        predictions['Electrical Conductance'],
    'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
})

output_path = os.path.join(OUTPUT_DIR, "submission_final.csv")
submission_df.to_csv(output_path, index=False)

print(f"\n✅ Submission saved: {output_path}")
