import pandas as pd
import xgboost as xgb

# === 1. 加载特征 ===
df = pd.read_csv("ml_data/ml_features_with_shape.csv")

# === 2. 明确与训练一致的特征列 ===
features = ['mean', 'std', 'min', 'max', 'range', 'median', 'slope', 'skew',
            'kurtosis', 'peak_val', 'peak_pos_rel', 'peak_value', 'peak_index',
            'duration', 'rising_slope', 'falling_slope', 'symmetry']

X = df[features]

# === 3. 加载模型 ===
model = xgb.XGBClassifier()
model.load_model("ml_output/xgb_model_full.json")

# === 4. 预测概率 ===
df['Prob'] = model.predict_proba(X)[:, 1]

# === 5. 输出中心年份 + 概率
output = df[['center_year', 'Prob']].rename(columns={'center_year': 'Year'})
output.to_csv("results/ml_prob_full_output.csv", index=False)
print("✅ Prediction complete! Saved to: results/ml_prob_full_output.csv")