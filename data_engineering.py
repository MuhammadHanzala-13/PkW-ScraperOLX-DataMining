"""
OLX Cars Pakistan - Data Engineering Pipeline
Input:  data/olx_cars_raw.csv    (raw scraped data)
Output: data/olx_cars_processed.csv (clean, feature-engineered data for modeling)

Pipeline Steps:
  1. Load raw data
  2. Rename columns to readable names
  3. Drop duplicate listings
  4. Clean and cast data types
  5. Drop rows with missing critical fields
  6. Prune price/mileage outliers (domain-based thresholds)
  7. Engineer new features
  8. Encode categorical columns
  9. Save processed CSV
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_FILE  = "data/olx_cars_raw.csv"
OUTPUT_FILE = "data/olx_cars_processed.csv"
CURRENT_YEAR = 2024

# ---------------------------------------------------------------------------
# STEP 1: Load Raw Data
# ---------------------------------------------------------------------------
def load_raw(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data not found: {filepath}. Run olx_scraper.py first.")
    df = pd.read_csv(filepath)
    print(f"[LOAD] {len(df)} records, {df.shape[1]} columns loaded from {filepath}.")
    return df

# ---------------------------------------------------------------------------
# STEP 2: Normalize Column Names
# OLX embeds params with arbitrary keys — we map known ones to clean names
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "price":       "price_raw",
    "location":    "city",
    "year":        "year",           # from params
    "mileage":     "mileage_km",     # from params
    "fueltype":    "fuel_type",      # from params
    "make":        "brand",          # from params
    "model":       "model",          # from params
    "color":       "color",          # from params
    "transmission": "transmission",  # from params
    "engine capacity": "engine_cc",  # from params
    "created_at":  "posted_date",
}

def normalize_columns(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Apply known renames where the column exists
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    print(f"[NORMALIZE] Columns after normalization: {list(df.columns)}")
    return df

# ---------------------------------------------------------------------------
# STEP 3: Drop Duplicates
# ---------------------------------------------------------------------------
def drop_duplicates(df):
    before = len(df)
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    print(f"[DEDUP] Removed {before - len(df)} duplicate records. Remaining: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# STEP 4: Clean & Cast Types
# ---------------------------------------------------------------------------
def clean_types(df):
    # Price: extract numeric value from raw string like "PKR 1,500,000"
    if "price_raw" in df.columns:
        df["price"] = (
            df["price_raw"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", np.nan)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Year: numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Mileage: numeric (strip "km", commas)
    if "mileage_km" in df.columns:
        df["mileage_km"] = (
            df["mileage_km"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", np.nan)
        )
        df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

    # Engine CC: numeric
    if "engine_cc" in df.columns:
        df["engine_cc"] = pd.to_numeric(
            df["engine_cc"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        )

    print(f"[TYPES] Data types corrected.")
    return df

# ---------------------------------------------------------------------------
# STEP 5: Drop Rows with Missing Critical Fields
# ---------------------------------------------------------------------------
CRITICAL_COLUMNS = ["price", "year"]

def drop_missing(df):
    before = len(df)
    # Only drop on columns that actually exist
    critical_present = [c for c in CRITICAL_COLUMNS if c in df.columns]
    df.dropna(subset=critical_present, inplace=True)
    print(f"[MISSING] Dropped {before - len(df)} rows with missing critical fields. Remaining: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# STEP 6: Outlier Pruning (Domain-based hard thresholds)
# ---------------------------------------------------------------------------
def prune_outliers(df):
    before = len(df)
    if "price" in df.columns:
        # Cars below Rs. 1 Lac or above Rs. 5 Crore are data errors or special cases
        df = df[(df["price"] >= 100_000) & (df["price"] <= 50_000_000)]

    if "year" in df.columns:
        # No car older than 1970 or from the future
        df = df[(df["year"] >= 1970) & (df["year"] <= CURRENT_YEAR)]

    if "mileage_km" in df.columns:
        # A car with 0km or over 1M km is unusable data
        df = df[(df["mileage_km"] > 0) & (df["mileage_km"] <= 1_000_000)]

    print(f"[OUTLIERS] Removed {before - len(df)} outlier rows. Remaining: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# STEP 7: Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    # 7a. Car Age (more linear relationship with price than year)
    if "year" in df.columns:
        df["car_age"] = CURRENT_YEAR - df["year"]

    # 7b. Log Price (right-skewed distribution correction for regression targets)
    if "price" in df.columns:
        df["price_log"] = np.log1p(df["price"])

    # 7c. Mileage per Year (normalized wear indicator)
    if "mileage_km" in df.columns and "car_age" in df.columns:
        df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)

    # 7d. High Mileage Flag (binary: driven hard or not)
    if "mileage_km" in df.columns:
        df["is_high_mileage"] = (df["mileage_km"] > 150_000).astype(int)

    print(f"[FE] Engineered features: car_age, price_log, mileage_per_year, is_high_mileage")
    return df

# ---------------------------------------------------------------------------
# STEP 8: Categorical Encoding
# ---------------------------------------------------------------------------
def encode_categoricals(df):
    # Label encoding for low-cardinality string columns
    for col in ["fuel_type", "transmission"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            df[f"{col}_encoded"] = pd.Categorical(df[col]).codes

    # Brand: keep top 15, group rest as "Other"
    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip().str.title()
        top_brands = df["brand"].value_counts().nlargest(15).index
        df["brand_clean"] = df["brand"].where(df["brand"].isin(top_brands), other="Other")

    print(f"[ENCODE] Categorical columns encoded.")
    return df

# ---------------------------------------------------------------------------
# STEP 9: Select Final Output Columns & Save
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "id", "title", "price", "price_log",
    "year", "car_age", "mileage_km", "mileage_per_year", "is_high_mileage",
    "brand", "brand_clean", "model", "fuel_type", "fuel_type_encoded",
    "transmission", "transmission_encoded", "engine_cc",
    "color", "city", "description", "posted_date",
]

def save_processed(df, filepath):
    os.makedirs("data", exist_ok=True)
    # Only keep columns that actually exist in the dataframe
    cols = [c for c in FINAL_COLUMNS if c in df.columns]
    df[cols].to_csv(filepath, index=False)
    print(f"[SAVE] {len(df)} processed records saved to {filepath}")
    print(f"[SAVE] Final columns: {cols}")

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("  OLX Cars | Data Engineering Pipeline")
    print("=" * 60)

    df = load_raw(INPUT_FILE)
    df = normalize_columns(df)
    df = drop_duplicates(df)
    df = clean_types(df)
    df = drop_missing(df)
    df = prune_outliers(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_processed(df, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("  Pipeline Complete.")
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
