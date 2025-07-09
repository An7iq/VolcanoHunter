# scripts/extract_features_from_windows.py
import pandas as pd
import numpy as np
import os

INPUT_PATH = "ml_data/ml_windows_labeled.csv"
OUTPUT_PATH = "ml_data/ml_features.csv"
os.makedirs("ml_data", exist_ok=True)

def extract_features_from_seq(seq_str):
    values = np.array([float(x) for x in seq_str.split(",")])

    # === 基础统计特征 ===
    mean_val = np.mean(values)
    std_val = np.std(values)
    max_val = np.max(values)
    min_val = np.min(values)
    range_val = max_val - min_val
    median_val = np.median(values)

    # === 形状类特征 ===
    slope = np.polyfit(np.arange(len(values)), values, deg=1)[0]
    skewness = pd.Series(values).skew()
    kurtosis = pd.Series(values).kurt()

    # === 峰值类特征 ===
    peak_idx = np.argmax(values)
    peak_val = values[peak_idx]
    peak_relative_pos = peak_idx / len(values)

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "median": median_val,
        "slope": slope,
        "skew": skewness,
        "kurtosis": kurtosis,
        "peak_val": peak_val,
        "peak_pos_rel": peak_relative_pos
    }

# === 加载窗口数据 ===
df = pd.read_csv(INPUT_PATH)

features = df["signal_seq"].apply(extract_features_from_seq)
features_df = pd.DataFrame(features.tolist())
features_df["center_year"] = df["center_year"]
features_df["label"] = df["label"]

features_df.to_csv(OUTPUT_PATH, index=False)
print(f"[✅ DONE] 特征文件保存至 {OUTPUT_PATH}，共 {len(features_df)} 条样本")