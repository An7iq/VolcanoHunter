import pandas as pd
import os
import re

DATA_RAW_DIR = os.path.join("data_raw")
DATA_CLEAN_DIR = os.path.join("data_clean")
os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

path = os.path.join(DATA_RAW_DIR, "edc2004sulfate.txt")

# === Step 1: 自动定位表头行 ===
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header_line_idx = None
for i, line in enumerate(lines):
    if "Age_yrbp" in line and "Sulfate_ugperL" in line:
        header_line_idx = i
        break

if header_line_idx is None:
    raise ValueError("❌ 未找到包含 'Age_yrbp' 的表头行！")

# === Step 2: 从 header_line_idx 开始重新读取 ===
df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=header_line_idx)
print("📄 原始列:", df.columns.tolist())

# 年份转换（BP → CE）
df["year_CE"] = 1950 - df["Age_yrbp"]

# 计算 sulfate deposition (ug/m²/year)
df["sulfate_deposition_ugm2yr"] = df["Sulfate_ugperL"] * df["Accrate_kgm-2yr-1"]

# 精简字段
df_clean = df[["year_CE", "Sulfate_ugperL", "Accrate_kgm-2yr-1", "sulfate_deposition_ugm2yr"]]
output_path = os.path.join(DATA_CLEAN_DIR, "edc_traditional_ready.csv")
df_clean.to_csv(output_path, index=False)

print(f"✅ 清洗后保存到: {output_path}")