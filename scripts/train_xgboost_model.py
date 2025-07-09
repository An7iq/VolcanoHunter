import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import json

# === 参数配置 ===
FULL_PATH = "ml_data/ml_windows_full_for_training.csv"
FILTERED_PATH = "ml_data/ml_windows_filtered_for_training.csv"
OUTPUT_DIR = "ml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 特征列设置（忽略非数值或元信息字段） ===
EXCLUDE_COLS = ["center_year", "signal_seq", "label"]

# === 保存分类报告 ===
def save_classification_report(report_dict, tag):
    df = pd.DataFrame(report_dict).T
    df.to_csv(f"{OUTPUT_DIR}/classification_report_{tag.lower()}.csv")
    with open(f"{OUTPUT_DIR}/classification_report_{tag.lower()}.json", "w") as f:
        json.dump(report_dict, f, indent=2)

# === 保存预测年份 ===
def save_predicted_years(df_test, y_pred, tag):
    detected_years = df_test.loc[y_pred == 1, "center_year"].values
    df_out = pd.DataFrame({"year": detected_years})
    df_out.to_csv(f"{OUTPUT_DIR}/predicted_volcano_years_{tag.lower()}.csv", index=False)
    print(f"📌 检测年份保存完成：predicted_volcano_years_{tag.lower()}.csv")

def train_and_evaluate(df, tag):
    df = df.dropna().reset_index(drop=True)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"[📊] {tag} 训练样本: {len(X_train)}，验证: {len(X_val)}，测试: {len(X_test)}")
    print("[DEBUG] All columns:", df.columns.tolist())
    print("[DEBUG] Selected feature columns:", feature_cols)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # === 评估 ===
    y_pred = model.predict(X_test)
    df_test = df.loc[idx_test].copy()
    df_test["y_pred"] = y_pred
    save_predicted_years(df_test, y_pred, tag)

    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    print(f"📌 [{tag}] 测试集评估结果：")
    print(classification_report(y_test, y_pred, digits=3))
    save_classification_report(report, tag)

    # === 混淆矩阵图 ===
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    plt.title(f"{tag} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{tag.lower()}.png", dpi=300)
    plt.close()

    # === 特征重要性图 ===
    xgb.plot_importance(model, max_num_features=10)
    plt.title(f"{tag} Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_{tag.lower()}.png", dpi=300)
    plt.close()

    # 保存模型
    model.save_model(f"{OUTPUT_DIR}/xgb_model_{tag.lower()}.json")
    print(f"✅ 模型和结果保存完成：{tag}")

    return pd.DataFrame(report).transpose()

# === 加载并训练两个版本 ===
report_full = train_and_evaluate(pd.read_csv(FULL_PATH), "FULL")
report_filtered = train_and_evaluate(pd.read_csv(FILTERED_PATH), "FILTERED")

# === 比较条形图绘制 ===
def plot_comparison_bar(report1, report2, label1="FULL", label2="FILTERED"):
    metrics = ["precision", "recall", "f1-score"]
    rows = ["macro avg", "weighted avg"]

    data = []
    for row in rows:
        data.append([report1.loc[row, m] for m in metrics])
        data.append([report2.loc[row, m] for m in metrics])

    labels = [f"{row} ({label})" for row in rows for label in [label1, label2]]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (row_data, label) in enumerate(zip(data, labels)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, row_data, width, label=label)

    ax.set_ylabel("Score")
    ax.set_title("FULL vs FILTERED Classification Report (macro & weighted avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_bar_chart.png", dpi=300)
    plt.close()
    print("📊 条形图已保存：comparison_bar_chart.png")

plot_comparison_bar(report_full, report_filtered)