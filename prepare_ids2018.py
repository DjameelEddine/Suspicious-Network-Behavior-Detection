"""
prepare_ids2018.py
==================
Prepares a sample of the CSE-CIC-IDS2018 dataset for real-time simulation.

Pipeline mirrors the exact preprocessing applied during training on CIC-IDS-2017:
  1.  Load CSVs & strip column names
  2.  Rename 2018 column names  ->  2017 naming convention
  3.  Drop duplicate column 'Fwd Header Length.1' (if present)
  4.  Drop duplicate rows
  5.  Fix label encoding  (2018 names -> 2017 names)
  6.  Filter to training classes only
  7.  Drop rows where Flow Duration == 0  (source of all Inf values)
  8.  Drop rows with integer-overflow negative values
  9.  Replace -1 sentinel in Init_Win columns with 0
  10. Drop zero-variance features  (same list as training)
  11. Log1p transform               (same feature list as training)
  12. Drop highly correlated features (same list as training)
  13. Keep only the 30 RF-selected features
  14. Scale using the saved StandardScaler (manual mean/scale to handle
      the case where scaler was fitted before RF feature selection)
  15. Balanced sample per class (--per_class rows, default 300)
  16. Sort chronologically by Timestamp if column exists, else shuffle
  17. Save to CSV

Usage
-----
  python prepare_ids2018.py --data_dir /path/to/2018/csvs --models_dir ./saved_models

Required artefacts in --models_dir
------------------------------------
  scaler.joblib            - fitted StandardScaler (from training notebook)
  selected_features.json   - list of 30 feature names after RF selection
  correlated_to_drop.json  - features removed by correlation filter
  log1p_features.json      - features that received log1p transformation
  zero_var_cols.json       - zero-variance features dropped during training
"""

import argparse
import gc
import glob
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Training classes  (2017 names -- ground truth for the models)
# ---------------------------------------------------------------------------
TRAINING_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Heartbleed",
]

# Columns that must be non-negative (overflow negatives -> drop row)
OVERFLOW_COLS = [
    "Flow Duration",
    "Flow Packets/s",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow Bytes/s",
]

# ---------------------------------------------------------------------------
# Column rename map:  2018 abbreviated names  ->  2017 full names
# Every one of the 30 selected features is covered here.
# ---------------------------------------------------------------------------
RENAME_2018_TO_2017 = {
    # Packet counts
    "Tot Fwd Pkts":             "Total Fwd Packets",
    "Tot Bwd Pkts":             "Total Backward Packets",
    "TotLen Fwd Pkts":          "Total Length of Fwd Packets",
    "TotLen Bwd Pkts":          "Total Length of Bwd Packets",
    # IAT totals  (2018 = 'Tot',  2017 = 'Total')
    "Fwd IAT Tot":              "Fwd IAT Total",
    "Bwd IAT Tot":              "Bwd IAT Total",
    # Packet lengths
    "Fwd Pkt Len Max":          "Fwd Packet Length Max",
    "Fwd Pkt Len Min":          "Fwd Packet Length Min",
    "Fwd Pkt Len Mean":         "Fwd Packet Length Mean",
    "Fwd Pkt Len Std":          "Fwd Packet Length Std",
    "Bwd Pkt Len Max":          "Bwd Packet Length Max",
    "Bwd Pkt Len Min":          "Bwd Packet Length Min",
    "Bwd Pkt Len Mean":         "Bwd Packet Length Mean",
    "Bwd Pkt Len Std":          "Bwd Packet Length Std",
    "Pkt Len Min":              "Min Packet Length",
    "Pkt Len Max":              "Max Packet Length",
    "Pkt Len Mean":             "Packet Length Mean",
    "Pkt Len Std":              "Packet Length Std",
    "Pkt Len Var":              "Packet Length Variance",
    "Pkt Size Avg":             "Average Packet Size",
    # Flow rates
    "Flow Byts/s":              "Flow Bytes/s",
    "Flow Pkts/s":              "Flow Packets/s",
    "Fwd Pkts/s":               "Fwd Packets/s",
    "Bwd Pkts/s":               "Bwd Packets/s",
    # Header lengths
    "Fwd Header Len":           "Fwd Header Length",
    "Bwd Header Len":           "Bwd Header Length",
    # Flag counts
    "FIN Flag Cnt":             "FIN Flag Count",
    "SYN Flag Cnt":             "SYN Flag Count",
    "RST Flag Cnt":             "RST Flag Count",
    "PSH Flag Cnt":             "PSH Flag Count",
    "ACK Flag Cnt":             "ACK Flag Count",
    "URG Flag Cnt":             "URG Flag Count",
    "CWE Flag Count":           "CWE Flag Count",
    "ECE Flag Cnt":             "ECE Flag Count",
    # Segment sizes
    "Fwd Seg Size Avg":         "Avg Fwd Segment Size",
    "Bwd Seg Size Avg":         "Avg Bwd Segment Size",
    "Fwd Seg Size Min":         "min_seg_size_forward",
    # Init window bytes
    "Init Fwd Win Byts":        "Init_Win_bytes_forward",
    "Init Bwd Win Byts":        "Init_Win_bytes_backward",
    # Active / Idle  (same names, listed for completeness)
    "Active Mean":              "Active Mean",
    "Active Std":               "Active Std",
    "Active Max":               "Active Max",
    "Active Min":               "Active Min",
    "Idle Mean":                "Idle Mean",
    "Idle Std":                 "Idle Std",
    "Idle Max":                 "Idle Max",
    "Idle Min":                 "Idle Min",
    # Subflows
    "Subflow Fwd Pkts":         "Subflow Fwd Packets",
    "Subflow Fwd Byts":         "Subflow Fwd Bytes",
    "Subflow Bwd Pkts":         "Subflow Bwd Packets",
    "Subflow Bwd Byts":         "Subflow Bwd Bytes",
    # Misc
    "Fwd Act Data Pkts":        "act_data_pkt_fwd",
    "Fwd Byts/b Avg":           "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg":           "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":         "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg":           "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":           "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg":         "Bwd Avg Bulk Rate",
    "Dst Port":                 "Destination Port",
    "Src Port":                 "Source Port",
}

# ---------------------------------------------------------------------------
# Label rename map:  2018 attack names  ->  2017 class names used in training
# ---------------------------------------------------------------------------
LABEL_2018_TO_2017 = {
    "Benign":                   "BENIGN",
    "DoS attacks-Hulk":         "DoS Hulk",
    "DoS attacks-GoldenEye":    "DoS GoldenEye",
    "DoS attacks-Slowloris":    "DoS slowloris",
    "DoS attacks-SlowHTTPTest": "DoS Slowhttptest",
    "Dos attacks-SlowHTTPTest": "DoS Slowhttptest",
    "DDOS attack-HOIC":         "DDoS",
    "DDoS attacks-LOIC-HTTP":   "DDoS",
    "DDOS attack-LOIC-UDP":     "DDoS",
    "FTP-BruteForce":           "FTP-Patator",
    "SSH-Bruteforce":           "SSH-Patator",
    "Brute Force -Web":         "Web Attack - Brute Force",
    "Brute Force -XSS":         "Web Attack - XSS",
    "SQL Injection":            "Web Attack - Sql Injection",
    "Infilteration":            "Infiltration",
    "Bot":                      "Bot",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path, fallback=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return fallback


def fix_labels(series):
    return series.str.strip().replace(LABEL_2018_TO_2017)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    models_dir = args.models_dir
    data_dir   = args.data_dir
    output     = args.output
    per_class  = args.per_class

    print("=" * 65)
    print("  CIC-IDS-2018  ->  Simulation Data Preparation")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 0. Load training artefacts
    # ------------------------------------------------------------------
    scaler_path   = os.path.join(models_dir, "scaler.joblib")
    sel_feat_path = os.path.join(models_dir, "selected_features.json")
    corr_path     = os.path.join(models_dir, "correlated_to_drop.json")
    log1p_path    = os.path.join(models_dir, "log1p_features.json")
    zvar_path     = os.path.join(models_dir, "zero_var_cols.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"scaler.joblib not found in '{models_dir}'.\n"
            "Run save_artefacts.py inside your Colab notebook first."
        )
    if not os.path.exists(sel_feat_path):
        raise FileNotFoundError(
            f"selected_features.json not found in '{models_dir}'.\n"
            "Run save_artefacts.py inside your Colab notebook first."
        )

    scaler            = joblib.load(scaler_path)
    selected_features = load_json(sel_feat_path)
    corr_to_drop      = load_json(corr_path,  fallback=[])
    log1p_features    = load_json(log1p_path, fallback=None)
    zero_var_saved    = load_json(zvar_path,  fallback=None)

    print(f"\n  Scaler              : {scaler_path}")
    print(f"  Selected features   : {len(selected_features)} features")
    print(f"  Corr cols to drop   : {len(corr_to_drop)}")
    print(f"  log1p features      : {'loaded' if log1p_features else 'will recompute'}")
    print(f"  Zero-var cols       : {'loaded' if zero_var_saved  else 'will recompute'}\n")

    # Build scaler lookup: feature name -> index in scaler.mean_ / scaler.scale_
    # Handles the case where the scaler was fitted before RF feature selection
    # (i.e. it knows more features than our 30).
    if hasattr(scaler, "feature_names_in_"):
        scaler_index = {f: i for i, f in enumerate(scaler.feature_names_in_)}
    else:
        scaler_index = {f: i for i, f in enumerate(selected_features)}

    # ------------------------------------------------------------------
    # 1. Load CSVs
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

    print(f"Found {len(csv_files)} CSV file(s). Loading ...")
    df_list = []
    for f in csv_files:
        tmp = pd.read_csv(f, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        df_list.append(tmp)
        print(f"  {os.path.basename(f):<55}  {tmp.shape}")

    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()
    print(f"\nShape after concat : {df.shape}")

    # ------------------------------------------------------------------
    # 2. Rename 2018 columns -> 2017 names
    # ------------------------------------------------------------------
    cols_renamed = {k: v for k, v in RENAME_2018_TO_2017.items() if k in df.columns}
    df.rename(columns=cols_renamed, inplace=True)
    print(f"Renamed {len(cols_renamed)} columns to match 2017 training names")

    # Preserve Timestamp for chronological sort later
    has_timestamp = "Timestamp" in df.columns
    timestamp_col = df["Timestamp"].copy() if has_timestamp else None

    # ------------------------------------------------------------------
    # 3. Drop duplicate Fwd Header Length column (2017 dataset artefact)
    # ------------------------------------------------------------------
    if "Fwd Header Length.1" in df.columns:
        df.drop(columns=["Fwd Header Length.1"], inplace=True)
        print("Dropped 'Fwd Header Length.1'")

    # ------------------------------------------------------------------
    # 4. Drop duplicate rows
    # ------------------------------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df):,} duplicate rows  ->  {len(df):,} remain")

    # ------------------------------------------------------------------
    # 5. Fix label encoding  (2018 names -> 2017 names)
    # ------------------------------------------------------------------
    df["Label"] = fix_labels(df["Label"])

    # ------------------------------------------------------------------
    # 6. Filter to training classes only
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df["Label"].isin(TRAINING_CLASSES)].copy()
    dropped = before - len(df)
    print(f"Filtered to training classes  ->  {len(df):,} rows"
          + (f"  (dropped {dropped:,} unlabelled/unknown)" if dropped else ""))

    print("\nClass distribution after filter:")
    for lbl, cnt in df["Label"].value_counts().items():
        print(f"  {lbl:<45} {cnt:>8,}")

    # ------------------------------------------------------------------
    # 7. Drop rows where Flow Duration == 0  (root cause of Inf values)
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df["Flow Duration"] != 0].copy()
    print(f"\nDropped {before - len(df):,} rows with Flow Duration == 0")

    # ------------------------------------------------------------------
    # 8. Drop integer-overflow negative rows
    # ------------------------------------------------------------------
    before = len(df)
    for col in OVERFLOW_COLS:
        if col in df.columns:
            df = df[df[col] >= 0]
    print(f"Dropped {before - len(df):,} rows with overflow negatives")

    # ------------------------------------------------------------------
    # 9. Replace -1 sentinel in Init_Win columns with 0
    #    (-1 = no TCP window observed, not a true negative)
    # ------------------------------------------------------------------
    for col in ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]:
        if col in df.columns:
            n = (df[col] == -1).sum()
            if n:
                df[col] = df[col].replace(-1, 0)
                print(f"Replaced {n:,} sentinel -1 in '{col}' with 0")

    # ------------------------------------------------------------------
    # 10. Drop zero-variance features
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if zero_var_saved is not None:
        zvar_present = [c for c in zero_var_saved if c in df.columns]
        df.drop(columns=zvar_present, inplace=True, errors="ignore")
        print(f"Dropped {len(zvar_present)} zero-variance cols (from training artefact)")
    else:
        zvar = [c for c in numeric_cols if df[c].std() == 0]
        df.drop(columns=zvar, inplace=True)
        print(f"Dropped {len(zvar)} zero-variance cols (recomputed)")

    # ------------------------------------------------------------------
    # 11. Log1p transformation
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if log1p_features is not None:
        to_transform = [f for f in log1p_features if f in df.columns]
    else:
        skew_s = df[numeric_cols].skew()
        min_s  = df[numeric_cols].min()
        to_transform = skew_s[(skew_s > 1) & (min_s >= 0)].index.tolist()
        print(f"  (Recomputed log1p candidate list: {len(to_transform)} features)")

    df[to_transform] = df[to_transform].apply(np.log1p)
    print(f"Applied log1p to {len(to_transform)} features")

    # ------------------------------------------------------------------
    # 12. Drop highly correlated features  (same list as training)
    # ------------------------------------------------------------------
    if corr_to_drop:
        present = [c for c in corr_to_drop if c in df.columns]
        df.drop(columns=present, inplace=True, errors="ignore")
        print(f"Dropped {len(present)} correlated features (from training artefact)")

    # ------------------------------------------------------------------
    # 13. Keep only the 30 RF-selected features  +  Label
    # ------------------------------------------------------------------
    missing_feats = [f for f in selected_features if f not in df.columns]
    if missing_feats:
        print(f"\nWARNING: {len(missing_feats)} selected feature(s) not found -- filling with 0:")
        for mf in missing_feats:
            print(f"  - {mf}")
            df[mf] = 0.0

    df = df[selected_features + ["Label"]].copy()
    print(f"\nShape after feature selection : {df.shape}")

    # ------------------------------------------------------------------
    # 14. Scale using the saved StandardScaler
    #
    #     Applied manually via  z = (x - mean) / scale  indexed by feature
    #     name.  This handles the case where the scaler was fitted on a
    #     wider set of features (before RF selection) without triggering
    #     sklearn's strict feature-name validation.
    # ------------------------------------------------------------------
    cols_scaled   = [f for f in selected_features if f in scaler_index]
    cols_unscaled = [f for f in selected_features if f not in scaler_index]

    print(f"\nScaler covers {len(cols_scaled)}/30 selected features")
    if cols_unscaled:
        print(f"  Not covered by scaler (left as-is): {cols_unscaled}")

    X = df[selected_features].copy()
    for col in cols_scaled:
        idx = scaler_index[col]
        X[col] = (X[col] - scaler.mean_[idx]) / scaler.scale_[idx]

    df[selected_features] = X
    del X
    gc.collect()
    print("Scaling applied")

    # ------------------------------------------------------------------
    # 15. Balanced sample per class
    # ------------------------------------------------------------------
    print(f"\nSampling up to {per_class} rows per class ...")
    sampled_parts = []
    for lbl in TRAINING_CLASSES:
        subset = df[df["Label"] == lbl]
        if len(subset) == 0:
            print(f"  (skipping '{lbl}' -- not present in 2018 data)")
            continue
        n = min(per_class, len(subset))
        sampled_parts.append(subset.sample(n=n, random_state=42))
        print(f"  {lbl:<45} {n:>5,} / {len(subset):>8,}")

    df_sim = pd.concat(sampled_parts, ignore_index=True)

    # ------------------------------------------------------------------
    # 16. Sort chronologically or shuffle
    # ------------------------------------------------------------------
    if has_timestamp and timestamp_col is not None:
        ts_aligned = timestamp_col.reindex(df_sim.index)
        df_sim["Timestamp"] = ts_aligned.values
        try:
            df_sim["Timestamp"] = pd.to_datetime(df_sim["Timestamp"])
            df_sim.sort_values("Timestamp", inplace=True)
            print("\nSorted chronologically by Timestamp")
        except Exception:
            df_sim.drop(columns=["Timestamp"], inplace=True, errors="ignore")
            df_sim = df_sim.sample(frac=1, random_state=42)
            print("\nTimestamp could not be parsed -- rows shuffled")
    else:
        df_sim = df_sim.sample(frac=1, random_state=42)
        print("\nNo Timestamp column -- rows shuffled (mixed traffic order)")

    df_sim.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 17. Save
    # ------------------------------------------------------------------
    df_sim.to_csv(output, index=False)

    print(f"\nSaved {len(df_sim):,} rows  ->  '{output}'")
    print("\nFinal class distribution:")
    for lbl, cnt in df_sim["Label"].value_counts().items():
        pct = cnt / len(df_sim) * 100
        print(f"  {lbl:<45} {cnt:>5,}  ({pct:.1f}%)")

    print(f"\nColumns : {list(df_sim.columns)}")
    print("\nDone")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class Args:
        data_dir = "new_data"
        models_dir = "saved_models"
        output = "simulation_data.csv"
        per_class = 300
    
    args = Args()
    main(args)