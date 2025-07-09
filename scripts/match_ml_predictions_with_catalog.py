# scripts/match_ml_predictions_with_catalog.py

import pandas as pd

ML_PATH = "ml_output/predicted_volcano_years_full.csv"
CATALOG_PATH = "volcano_catalogs/benchmark_volcano_years_full.csv"
OUTPUT_PATH = "results/ml_predictions_with_match_flag.csv"
YEAR_TOLERANCE = 3

# === 加载数据 ===
ml_df = pd.read_csv(ML_PATH)
catalog_df = pd.read_csv(CATALOG_PATH)

# === 兼容只有一列的情况 ===
ml_col = ml_df.columns[0]
catalog_col = catalog_df.columns[0]

ml_df = ml_df.dropna(subset=[ml_col])
catalog_df = catalog_df.dropna(subset=[catalog_col])

ml_df[ml_col] = ml_df[ml_col].astype(int)
catalog_years = set(catalog_df[catalog_col].astype(int).tolist())

def find_match(year):
    for offset in range(-YEAR_TOLERANCE, YEAR_TOLERANCE + 1):
        if (year + offset) in catalog_years:
            return year + offset
    return None

ml_df['Matched_Year'] = ml_df[ml_col].apply(find_match)
ml_df['Matched'] = ml_df['Matched_Year'].notnull()
ml_df['Notes'] = ml_df['Matched'].apply(lambda x: 'Known Event' if x else 'Potential New Event')

ml_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Matching complete! Saved to: {OUTPUT_PATH}")