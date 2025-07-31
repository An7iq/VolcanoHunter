# Volcano Hunter v1.0

Volcano Hunter is a machine learning pipeline for detecting volcanic eruptions in ice core sulfate records. This repository provides a simplified, reproducible version of the workflow, aimed at demonstrating the core structure of the model using publicly available data.



## Overview

This pipeline processes raw ice core chemistry data, constructs sliding windows around each year, extracts morphological features, and trains an XGBoost classifier to identify volcanic events. It then compares predicted events with a benchmark catalog of known eruptions.



## Directory Structure

```
volcano-hunter/
├── data_raw/                      # Original EDC data & GVP catalog
├── scripts/                       # Core processing & ML scripts
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and instructions
```



## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare EDC dataset

```bash
python scripts/preprocess_EDC_for_ML.py
```

#### 📓 notebooks/01_EDC_data_preprocessing.ipynb

This notebook processes the Wolff et al. (2010) EDC sulfate dataset and exports a cleaned, annual-resolution CSV (EDC_merged_4ML.csv) for machine learning tasks. It performs:
	•	Column cleaning and renaming (e.g., nssSO₄)
	•	Handling of missing years
	•	Linear interpolation across gaps
	•	Alignment to continuous year range
	•	Export to CSV for ML model input

Output:
data/EDC_merged_4ML.csv

🧠 Tip for beginners: If you don’t understand what a line does, read the comments in the notebook. This is the only file needed to generate ML input data from the raw Wolff dataset.

### 3. Generate ML windows

```bash
python scripts/generate_ml_windows.py
```

### 4. Extract features

```bash
python scripts/extract_features_from_windows.py
```

### 5. Train the model

```bash
python scripts/train_xgboost_model.py
```

### 6. Predict and match with catalog

```bash
python scripts/predict_prob_from_saved_model.py
python scripts/match_ml_predictions_with_catalog.py
```



## Data

- `wolff2010-edc-ions-aicc2012.txt`: EDC ice core chemistry (nssSO₄)
- `GVP_Volcano_List_Holocene.csv`: Global Volcanism Program reference catalog

Note: Only minimal raw data is included to demonstrate pipeline functionality.



## Citation

If you use this project or find it helpful, please cite relevant ice core and volcanism datasets such as:

- Sigl et al. (2015), *Nature*
- Toohey et al. (2021), *Earth System Science Data*
- Global Volcanism Program (GVP)
