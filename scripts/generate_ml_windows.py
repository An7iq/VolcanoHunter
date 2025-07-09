# scripts/generate_ml_windows.py
import pandas as pd
import numpy as np
import os

# ==== 参数 ====
INPUT_CSV = "data_clean/EDC_merged_4ML.csv"  # 已处理好的主数据
# ⚠️ 训练阶段使用全量 benchmark，包含部分潜在但不确定事件
# 评估阶段将单独使用 benchmark_volcano_years_with_sources.csv（高可信事件）
# VOLCANO_CATALOG = "volcano_catalogs/benchmark_volcano_years_with_sources.csv"
VOLCANO_CATALOG = "volcano_catalogs/benchmark_volcano_years_full.csv"
OUTPUT_CSV = "ml_data/ml_windows_labeled.csv"
os.makedirs("ml_data", exist_ok=True)

WINDOW_SIZE = 20      # 每个窗口长度，单位：年
STRIDE = 1            # 滑动步长
LABEL_TOLERANCE = 3   # 容许事件年份偏差 ±N 年

# ==== Step 1: 加载数据 ====
df = pd.read_csv(INPUT_CSV)
df = df.rename(columns={"nssSO4flux_trad_ugm-2yr-1": "signal"})
df["year_CE"] = (2000 - df["b2k_age"]).round().astype(int)
df = df[["year_CE", "signal"]].dropna()

# ==== Step 2: 加载 benchmark 火山年份 ====
volcano_df = pd.read_csv(VOLCANO_CATALOG)
volcano_years = set(volcano_df["year_CE"].round().astype(int))

# ==== Step 3: 构建窗口序列并打标签 ====
data = []

years = df["year_CE"].values
signals = df["signal"].values

for i in range(len(df) - WINDOW_SIZE + 1):
    window_years = years[i:i+WINDOW_SIZE]
    window_signal = signals[i:i+WINDOW_SIZE]
    center_year = int(np.mean(window_years))

    # 事件标注：±LABEL_TOLERANCE 内是否有火山事件
    has_event = any(abs(center_year - vy) <= LABEL_TOLERANCE for vy in volcano_years)

    data.append({
        "center_year": center_year,
        "signal_seq": ",".join([str(v) for v in window_signal]),
        "label": int(has_event)
    })

df_out = pd.DataFrame(data)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"[✅ Done] 输出训练窗口序列，共 {len(df_out)} 个样本，保存至：{OUTPUT_CSV}")