"""
TerraClimate Multi-Variable Extraction
=======================================
Extracts climate variables from the TerraClimate dataset via the
Microsoft Planetary Computer API for use in water quality prediction.

TerraClimate provides monthly climate data at ~4 km resolution dating
back to 1958. This script extracts variables for specific sampling
locations and dates, matching each location to the nearest grid point.

Variables extracted:
  - ppt   : Monthly precipitation accumulation (mm)
  - soil  : Soil moisture (mm)
  - def   : Climate water deficit (mm)
  - tmax  : Maximum temperature (°C × 10)
  - q     : Runoff (mm)

These climate features significantly improve water quality prediction
because they generalize across geographic regions — unlike raw satellite
band values which are location-specific.

The script reconnects to the Planetary Computer before each variable
to avoid authentication token expiry (~45 min lifetime).

Usage:
  python extract_terraclimate.py

Output:
  data/terraclimate_training_multi.csv
  data/terraclimate_validation_multi.csv

Requirements:
  pip install xarray pystac-client planetary-computer scipy tqdm zarr adlfs dask
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer as pc
from tqdm import tqdm
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

# Variables to extract — chosen for relevance to water quality:
#   ppt  → rainfall drives runoff, dilution, pollution transport
#   soil → saturated soil means more surface runoff into rivers
#   def  → drought concentrates pollutants in low water flow
#   tmax → temperature affects evaporation and biological activity
#   q    → runoff directly measures water entering rivers
VARIABLES_TO_EXTRACT = ['ppt', 'soil', 'def', 'tmax', 'q']

# ============================================================
# API FUNCTIONS
# ============================================================

def load_terraclimate_dataset():
    """
    Open a fresh connection to the TerraClimate dataset on
    Microsoft Planetary Computer.

    Called before each variable to avoid token expiry.
    The Planetary Computer issues short-lived SAS tokens
    (~45 minutes) for Azure Blob Storage access. Long
    extractions exceed this window, so reconnecting ensures
    a fresh token.
    """
    print("   🌐 Connecting to Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )
    print("   ✅ Connected")
    return ds


def filter_south_africa(ds, var):
    """
    Filter global TerraClimate data to South Africa (2011–2015).

    The full dataset covers the entire globe at ~4 km resolution.
    We subset to the bounding box of South Africa and the 5-year
    study period, then convert to a flat DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        Full TerraClimate dataset.
    var : str
        Climate variable name (e.g. 'ppt', 'soil').

    Returns
    -------
    DataFrame with columns: Latitude, Longitude, Sample Date, {var}
    """
    ds_filtered = ds[var].sel(time=slice("2011-01-01", "2015-12-31"))

    frames = []
    for i in tqdm(range(len(ds_filtered.time)), desc=f"   {var}"):
        df = ds_filtered.isel(time=i).to_dataframe().reset_index()
        df = df[
            (df['lat'] > -35.18) & (df['lat'] < -21.72) &
            (df['lon'] > 14.97) & (df['lon'] < 32.79)
        ]
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result['time'] = result['time'].astype(str)
    result = result.rename(columns={
        "lat": "Latitude", "lon": "Longitude", "time": "Sample Date"
    })

    print(f"   ✅ {var}: {len(result):,} grid records")
    return result


def assign_nearest_climate(sample_df, climate_df, var_name):
    """
    Map each sampling location to its nearest TerraClimate grid point
    and assign the climate value for the closest available date.

    Uses a KD-tree for efficient nearest-neighbor lookup in geographic
    coordinates. For each sample, finds the nearest grid point, then
    selects the climate reading closest in time.

    Parameters
    ----------
    sample_df : DataFrame
        Sampling locations with Latitude, Longitude, Sample Date.
    climate_df : DataFrame
        Filtered TerraClimate grid data.
    var_name : str
        Climate variable name.

    Returns
    -------
    DataFrame with a single column {var_name} aligned to sample_df rows.
    """
    sa_coords = np.radians(sample_df[['Latitude', 'Longitude']].values)
    climate_coords = np.radians(climate_df[['Latitude', 'Longitude']].values)

    tree = cKDTree(climate_coords)
    _, idx = tree.query(sa_coords, k=1)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)

    sample_df = sample_df.reset_index(drop=True)
    sample_df[['nearest_lat', 'nearest_lon']] = nearest_points[['Latitude', 'Longitude']]

    sample_df['Sample Date'] = pd.to_datetime(
        sample_df['Sample Date'], dayfirst=True, errors='coerce'
    )
    climate_df['Sample Date'] = pd.to_datetime(
        climate_df['Sample Date'], dayfirst=True, errors='coerce'
    )

    values = []
    for i in tqdm(range(len(sample_df)), desc=f"   Mapping {var_name}"):
        sample_date = sample_df.loc[i, 'Sample Date']
        nlat = sample_df.loc[i, 'nearest_lat']
        nlon = sample_df.loc[i, 'nearest_lon']

        subset = climate_df[
            (climate_df['Latitude'] == nlat) &
            (climate_df['Longitude'] == nlon)
        ]

        if subset.empty:
            values.append(np.nan)
            continue

        nearest_idx = (subset['Sample Date'] - sample_date).abs().idxmin()
        values.append(subset.loc[nearest_idx, var_name])

    return pd.DataFrame({var_name: values})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    print("📂 Loading sample locations...")
    training_df = pd.read_csv(
        os.path.join(DATA_DIR, "water_quality_training_dataset.csv")
    )
    validation_df = pd.read_csv("submissions/submission_template.csv")
    print(f"   Training: {len(training_df):,} | Validation: {len(validation_df)}")

    # Initialize output DataFrames
    # If a partial extraction exists, load it and skip completed variables
    multi_train_path = os.path.join(DATA_DIR, "terraclimate_training_multi.csv")
    multi_val_path = os.path.join(DATA_DIR, "terraclimate_validation_multi.csv")

    if os.path.exists(multi_train_path):
        train_multi = pd.read_csv(multi_train_path)
        val_multi = pd.read_csv(multi_val_path)
        existing = [
            c for c in train_multi.columns
            if c not in ['Latitude', 'Longitude', 'Sample Date']
        ]
        print(f"   Existing variables: {existing}")
    else:
        train_multi = pd.DataFrame({
            'Latitude': training_df['Latitude'],
            'Longitude': training_df['Longitude'],
            'Sample Date': training_df['Sample Date']
        })
        val_multi = pd.DataFrame({
            'Latitude': validation_df['Latitude'],
            'Longitude': validation_df['Longitude'],
            'Sample Date': validation_df['Sample Date']
        })
        existing = []

    # Extract each variable
    for var in VARIABLES_TO_EXTRACT:
        if var in existing:
            print(f"\n⏭️  Skipping {var} (already extracted)")
            continue

        print(f"\n{'='*60}")
        print(f"📊 Extracting: {var}")
        print(f"{'='*60}")

        var_start = time.time()

        # Fresh API connection per variable (avoids token expiry)
        ds = load_terraclimate_dataset()

        # Filter global dataset to South Africa + 2011-2015
        tc_data = filter_south_africa(ds, var)

        # Map to training locations
        print(f"\n   Mapping to {len(training_df):,} training locations...")
        train_var = assign_nearest_climate(training_df.copy(), tc_data.copy(), var)
        train_multi[var] = train_var[var].values

        # Map to validation locations
        print(f"\n   Mapping to {len(validation_df)} validation locations...")
        val_var = assign_nearest_climate(validation_df.copy(), tc_data.copy(), var)
        val_multi[var] = val_var[var].values

        elapsed = (time.time() - var_start) / 60
        print(f"\n   ⏱️  {var} completed in {elapsed:.1f} minutes")

        # Save progress after each variable (resume-safe)
        train_multi.to_csv(multi_train_path, index=False)
        val_multi.to_csv(multi_val_path, index=False)

        current_vars = [
            c for c in train_multi.columns
            if c not in ['Latitude', 'Longitude', 'Sample Date']
        ]
        print(f"   💾 Saved — variables so far: {current_vars}")

    total = (time.time() - start_time) / 60
    print(f"\n🎉 Done in {total:.1f} minutes")
    print(f"   Output: {multi_train_path}")
    print(f"   Output: {multi_val_path}")
