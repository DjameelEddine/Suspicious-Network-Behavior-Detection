# 🛡️ IDS Real-Time Detection Dashboard

A real-time **two-layer machine learning pipeline** for network intrusion detection, built with **Streamlit**. This dashboard simulates live network traffic by processing rows from the CIC-IDS-2018 dataset sequentially, applying a trained binary classifier (Layer 1) followed by a multi-class attack-type classifier (Layer 2).

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Dashboard Features](#dashboard-features)
5. [Detection Pipeline](#detection-pipeline)
6. [Configuration](#configuration)
7. [File Structure](#file-structure)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What It Does

The dashboard provides **real-time visualization** of network intrusion detection using a trained ML model:

- **Layer 1 (Binary)**: Classifies each network flow as either **BENIGN** or **ATTACK**
  - Model options: XGBoost, Logistic Regression, Random Forest, KNN, Decision Tree
- **Layer 2 (Multi-class)**: For flows predicted as ATTACK, identifies the specific attack type
  - 15 attack types: DoS Hulk, DDoS, PortScan, FTP-Patator, SSH-Patator, Bot, Web Attack variants, etc.
  - Model options: KNN, XGBoost, Logistic Regression, Random Forest, Decision Tree

### Key Features

✅ **Real-time simulation** – Process rows one-by-one with configurable delays  
✅ **Two-layer detection** – Binary classification followed by attack-type identification  
✅ **Live dashboards** – KPIs, rolling attack rate chart, class distribution pie chart  
✅ **Alert feed** – Last 5 detected attacks with confidence scores  
✅ **Detection log** – Detailed table with Layer 1 & Layer 2 predictions vs. ground truth  
✅ **Model selection** – Choose between multiple pre-trained models per layer  
✅ **Session controls** – Start, pause, and reset simulation at any time  

---

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/Suspicious-Network-Behavior-Detection
```

### 2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas numpy joblib plotly scikit-learn xgboost
```

### 3. Prepare Data & Models

Before running the dashboard, you must:

1. **Save training artefacts** from your training notebook:
   ```
   saved_models/
   ├── binary/
   │   ├── xgboost_binary.joblib
   │   ├── knn_binary.joblib
   │   └── ... (other binary models)
   ├── multi/
   │   ├── knn_multi.joblib
   │   ├── xgboost_multi.joblib
   │   └── ... (other multi-class models)
   ├── scaler.joblib              ← StandardScaler fitted during training
   ├── selected_features.json     ← 30 feature names after RF selection
   ├── correlated_to_drop.json    ← Features removed by correlation filter
   ├── log1p_features.json        ← Features that received log1p
   └── zero_var_cols.json         ← Zero-variance columns dropped
   ```

2. **Prepare simulation data** from CIC-IDS-2018 dataset:
   ```bash
   python prepare_ids2018.py \
       --data_dir /path/to/CIC-IDS2018-csvs \
       --models_dir ./saved_models \
       --output simulation_data.csv \
       --per_class 300
   ```
   
   This creates `simulation_data.csv` with 300 balanced samples per class.

---

## Quick Start

### 1. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens in your browser at `http://localhost:8501`

### 2. Configure Paths (Sidebar)

- **Models directory**: Path to `saved_models/` folder  
  Default: `./saved_models`
- **Simulation CSV**: Path to `simulation_data.csv`  
  Default: `./testing_data/simulation_data.csv`

### 3. Select Models

- **Layer 1 – Binary model**: `xgboost_binary.joblib` (default)
- **Layer 2 – Multi-class model**: `knn_multi.joblib` (default)

### 4. Load Models & Data

Click **"🔄 Load Models & Data"** button. Wait for success message.

### 5. Configure Simulation

- **Interval (seconds)**: Delay between rows (1–30 sec)
- **Show BENIGN rows in log**: Toggle to include/exclude benign rows from the detection log

### 6. Run Simulation

- Click **"▶ Start"** to begin processing rows
- Click **"⏸ Pause"** to pause
- Click **"🔁 Reset Simulation"** to clear all data and start over

---

## Dashboard Features

### KPIs (Top Row)

```
Total Processed  │  Attacks Detected (with %)  │  Benign Flows  │  Attack Rate (%)
```

Real-time counters updating as rows are processed.

### Rolling Attack Rate Chart

📈 Line chart showing the **cumulative attack detection rate** over the last 50 rows.
- X-axis: Row index (0–50)
- Y-axis: Attack rate percentage (0–100%)
- Colour: Red (#e74c3c)

**What it shows:**
- As each new row arrives, the chart updates to show: **"What % of all rows so far were predicted as ATTACK?"**
- Starts at 100% if early rows are attacks, then **decreases** as benign traffic arrives
- Example: If rows [ATTACK, ATTACK, BENIGN, BENIGN], the rate goes 100% → 100% → 66.7% → 50%

**Use case:** Quickly spot traffic patterns – a sudden drop means benign traffic, a spike means attack cluster

### Class Distribution Pie Chart

 Donut chart showing the breakdown of attack types detected so far.
- Each slice represents an attack class (e.g., DDoS, PortScan, Bot)
- Colour-coded by attack type

### Alert Feed

🚨 Recently predicted rows (color-coded by correctness):
- Row index
- Layer 1 prediction badge (BENIGN or ATTACK)
- Layer 2 prediction badge (attack type or BENIGN)
- Confidence score of Layer 2 prediction
- Ground truth label
- **Dark Green**: Prediction was **correct**
- **Dark Red**: Prediction was **incorrect**

### Detection Log

📋 Scrollable table with last 30 rows (configurable):
- **Row**: Flow index
- **True Label**: Ground truth label from CSV
- **L1 Prediction**: Layer 1 binary prediction (BENIGN or ATTACK)
- **L1 Conf**: Layer 1 confidence (0–100%)
- **L2 Prediction**: Layer 2 attack type (or BENIGN)
- **L2 Conf**: Layer 2 confidence (0–100%)
- **L1 ✓**: Whether Layer 1 prediction was correct
- **L2 ✓**: Whether Layer 2 prediction was correct
- **Highlighting**:
  - **Dark Green**: L2 prediction was **correct** (matches true label)
  - **Dark Red**: L2 prediction was **incorrect** (doesn't match true label)

---

## Detection Pipeline

### Per-Row Processing

```
Network Flow Features (30 features)
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
    Layer 1: Binary Classifier         Layer 2: Multi-class Classifier
    (e.g., XGBoost)                    (e.g., KNN)
    Predicts: BENIGN or ATTACK         Predicts: Specific attack type
                │                              │
                ▼                              ▼
        P(BENIGN) vs P(ATTACK)      P(DoS) vs P(DDoS) vs P(PortScan) vs ...
                │                              │
                └──────────────────┬───────────┘
                                   │
                    Compare Confidences:
                   Layer 1 BENIGN P vs Layer 2 Max P
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Take HIGHER         │
                        │ Confidence Result   │
                        └─────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │                     │
         If L1 BENIGN > L2 Attack   If L2 Attack > L1 BENIGN
              Return BENIGN             Return ATTACK + Type
```

### How It Works

1. **Both layers always run independently** – Neither skips
2. **Layer 1** outputs: `P(BENIGN)` and `P(ATTACK)` = `1 - P(BENIGN)`
3. **Layer 2** outputs: Probabilities for all 15 attack types + BENIGN class
4. **Comparison**: 
   - Layer 1 BENIGN confidence vs Layer 2's maximum probability
   - Whichever is higher determines the final prediction
5. **Result**: 
   - If Layer 1 is more confident → **BENIGN**
   - If Layer 2 is more confident → **ATTACK** (specific type)

### Advantages of This Approach

✅ **No missed attacks** – Layer 2 can catch attacks even if Layer 1 is uncertain  
✅ **Fewer false alarms** – Layer 1 can confidently reject benign traffic  
✅ **Intelligent trade-off** – Picks the most confident layer automatically  
✅ **Debuggable** – Detection log shows both layers' predictions for analysis

### Confidence Calculation

**Layer 1 Confidence:**
- `P(BENIGN)` = `1 - P(ATTACK)` if final prediction is BENIGN
- `P(ATTACK)` if final prediction is ATTACK
- Source: `predict_proba()[0][1]` or fallback to `1.0`

**Layer 2 Confidence:**
- Maximum probability across all 15 attack classes
- Source: `max(predict_proba()[0])` or fallback to `1.0`

**Note:** Confidence values come **directly from the trained models**, not calculated by the dashboard. Each model learned these probabilities during training.

---

## Configuration

### Command-Line Arguments (for `prepare_ids2018.py`)

```bash
python prepare_ids2018.py \
    --data_dir /path/to/2018/csvs \
    --models_dir ./saved_models \
    --output simulation_data.csv \
    --per_class 300
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *(required)* | Folder containing CIC-IDS-2018 CSV files |
| `--models_dir` | *(required)* | Folder with `scaler.joblib` and JSON artefacts |
| `--output` | `simulation_data.csv` | Output CSV filename |
| `--per_class` | `300` | Max rows sampled per class |

### Dashboard Constants (in `dashboard.py`)

Edit these to customize behavior:

```python
DEFAULT_MODELS_DIR   = "./saved_models"      # Models folder path
DEFAULT_SIM_CSV      = "./testing_data/simulation_data.csv"  # Sim data path
DEFAULT_INTERVAL     = 5                     # Seconds between rows (default)
MAX_HISTORY          = 500                   # Max rows retained in history
ROLLING_WINDOW       = 50                    # Window size for rolling chart
```

---

## File Structure

```
project/
├── dashboard.py                    ← Streamlit dashboard (run this)
├── prepare_ids2018.py             ← Data preparation script
├── requirements.txt               ← Python dependencies
├── dashboard_readme.md            ← This file
│
├── saved_models/
│   ├── binary/                    ← Binary classifiers
│   │   ├── xgboost_binary.joblib
│   │   ├── knn_binary.joblib
│   │   └── ...
│   ├── multi/                     ← Multi-class classifiers
│   │   ├── knn_multi.joblib
│   │   ├── xgboost_multi.joblib
│   │   └── ...
│   ├── scaler.joblib              ← StandardScaler
│   ├── selected_features.json     ← 30 feature names
│   ├── correlated_to_drop.json    ← Correlation-dropped features
│   ├── log1p_features.json        ← Log1p-transformed features
│   └── zero_var_cols.json         ← Zero-variance columns
│
├── testing_data/ (or custom path)
│   └── simulation_data.csv        ← Prepared simulation data
│
└── data/  (optional)
    └── (raw CIC-IDS-2018 CSVs)
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'streamlit'`

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### Error: `FileNotFoundError: scaler.joblib not found`

**Solution:** Run `save_artefacts.py` in your training notebook to export the scaler and feature lists.

### Error: `File not found: ./testing_data/simulation_data.csv`

**Solution:** Run `prepare_ids2018.py` to prepare the simulation data:

```bash
python prepare_ids2018.py --data_dir /path/to/2018/csvs --models_dir ./saved_models
```

### Dashboard is slow

**Causes:**
- Interval too short – increase slider value (1–30 sec)
- Too many rows kept in history – reduce `MAX_HISTORY` in code

### Models won't load

**Checklist:**
- [ ] Paths in sidebar are correct
- [ ] Model files exist and end with `.joblib`
- [ ] Model files are in `binary/` and `multi/` subdirectories
- [ ] All required dependencies installed (`xgboost`, `scikit-learn`, etc.)

### Confidence scores always look wrong

**Remember:** Confidence comes **from the trained models**, not the dashboard. If models were poorly trained or overfitted, confidence scores will reflect that. Check model performance on test data before using in production.

---

## Next Steps

1. **Experiment with different model combinations** – Try XGBoost + KNN, RF + Logistic Regression, etc.
2. **Adjust simulation speed** – Speed up/down with the interval slider
3. **Analyze detection patterns** – Review the alert feed and detection log for insights
4. **Export results** – Modify the code to save predictions to a file for post-analysis
5. **Train on fresh data** – Use CIC-IDS-2017 or another benchmark dataset

---

## References

- **CIC-IDS-2017/2018**: Intrusion Detection Evaluation Datasets  
  https://www.unb.ca/cic/datasets/ids-2018.html
- **Streamlit Documentation**: https://docs.streamlit.io
- **scikit-learn**: https://scikit-learn.org
- **XGBoost**: https://xgboost.readthedocs.io

---

**Author**: IDS Project  
**Last Updated**: March 2026
