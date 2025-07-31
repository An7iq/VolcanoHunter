"""Preprocess the EDC sulfate dataset and export a machine-learning-ready CSV.

This script reads the Wolff et al. (2010) EDC ion concentration dataset,
cleans and interpolates the nss_SO4 record, aligns the data to annual resolution,
and saves the result as 'EDC_merged_4ML.csv' for use in ML-based detection models.
"""

import pandas as pd
import numpy as np
import os

# Set input and output paths
INPUT_PATH = "data/wolff2010-edc-ions-aicc2012.txt"
OUTPUT_PATH = "data/EDC_merged_4ML.csv"

# Load the original data
df_raw = pd.read_csv(INPUT_PATH, sep="\t", comment="#")

# Clean column names and select relevant data
df_raw.columns = df_raw.columns.str.strip()
df = df_raw.rename(columns={"nssSO4 (ppb)": "nss_SO4"})

# Drop rows with missing Year or nss_SO4
df = df[["Year", "nss_SO4"]].dropna()
df = df.sort_values(by="Year")

# Ensure Year is numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df = df.set_index("Year")

# Generate full year index
year_min, year_max = int(df.index.min()), int(df.index.max())
full_years = np.arange(year_min, year_max + 1)

# Reindex and interpolate missing values
df_interp = df.reindex(full_years)
df_interp["nss_SO4"] = df_interp["nss_SO4"].interpolate(method="linear")

# Reset index and save
df_final = df_interp.reset_index().rename(columns={"index": "Year"})
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"✅ EDC sulfate data preprocessed and saved to: {OUTPUT_PATH}")