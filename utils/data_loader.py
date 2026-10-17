"""
Universal Dataset Loaders for Cybersecurity Benchmark Datasets:
1. UNSW-NB15 (Network Anomaly & Attack Capture)
2. CSE-CIC-IDS2018 (Botnet & Intrusion Flow Capture)
"""

import os
import gc
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_unsw_nb15(data_dir=None):
    """
    Finds or downloads the official UNSW-NB15 dataset.
    Returns: X_train_scaled, y_train, X_test_scaled, y_test, feature_names
    """
    train_file = None
    test_file = None

    search_dirs = [
        data_dir,
        os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "mrwellsdavid", "unsw-nb15", "versions", "1"),
        os.path.join(os.path.expanduser("~"), "Downloads"),
        ".",
        ".."
    ]

    for d in filter(None, search_dirs):
        if os.path.exists(d):
            c_train = os.path.join(d, "UNSW_NB15_training-set.csv")
            c_test = os.path.join(d, "UNSW_NB15_testing-set.csv")
            if os.path.exists(c_train) and os.path.exists(c_test):
                train_file, test_file = c_train, c_test
                break

    if not train_file:
        print("[*] UNSW-NB15 dataset not found locally. Downloading via KaggleHub...", flush=True)
        try:
            import kagglehub
            path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
            train_file = os.path.join(path, "UNSW_NB15_training-set.csv")
            test_file = os.path.join(path, "UNSW_NB15_testing-set.csv")
        except Exception as e:
            raise RuntimeError(f"Failed to auto-download UNSW-NB15: {e}")

    print(f"[+] Loaded Train: {train_file}", flush=True)
    print(f"[+] Loaded Test : {test_file}", flush=True)

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Standardize column names
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # Drop leak & metadata columns
    drop_cols = ['id', 'attack_cat', 'label']
    y_train = df_train['label'].values.astype(np.int64)
    y_test = df_test['label'].values.astype(np.int64)

    X_train_df = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
    X_test_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

    # Encode categorical columns
    categorical_cols = [c for c in X_train_df.columns if not pd.api.types.is_numeric_dtype(X_train_df[c])]
    for col in categorical_cols:
        le = LabelEncoder()
        all_vals = list(X_train_df[col].astype(str)) + list(X_test_df[col].astype(str))
        le.fit(all_vals)
        X_train_df[col] = le.transform(X_train_df[col].astype(str))
        X_test_df[col] = le.transform(X_test_df[col].astype(str))

    feature_names = X_train_df.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df.values).astype(np.float32)

    del df_train, df_test, X_train_df, X_test_df
    gc.collect()

    return X_train_scaled, y_train, X_test_scaled, y_test, feature_names


def load_cicids2018(max_samples=250000, data_path=None):
    """
    Locates CSE-CIC-IDS2018 dataset (03-02-2018.csv or zip).
    Cleans infinite/NaN values and prepares stratified train-test splits.
    """
    candidates = [
        data_path,
        os.path.join(os.path.expanduser("~"), "Downloads", "03-02-2018.csv.zip"),
        os.path.join(os.path.expanduser("~"), "Downloads", "03-02-2018.csv"),
        "03-02-2018.csv.zip",
        "03-02-2018.csv",
        "../03-02-2018.csv.zip"
    ]

    target_path = None
    for p in filter(None, candidates):
        if os.path.exists(p):
            target_path = p
            break

    if not target_path:
        raise FileNotFoundError("Could not locate 03-02-2018.csv or 03-02-2018.csv.zip in search paths.")

    print(f"[+] Found Dataset File: {target_path} ({os.path.getsize(target_path)/(1024*1024):.2f} MB)", flush=True)
    t0 = time.time()

    if target_path.endswith(".zip"):
        with zipfile.ZipFile(target_path, 'r') as z:
            csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(target_path)

    print(f"[+] Loaded {len(df):,} total raw network flow records in {time.time() - t0:.2f}s", flush=True)

    df.columns = df.columns.str.strip()
    label_col = 'Label'
    y_raw = df[label_col].astype(str).str.strip()
    y_binary = (y_raw != 'Benign').astype(np.int64)

    drop_cols = ['Label', 'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port']
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    print("[*] Sanitizing Infinite and NaN flow values...", flush=True)
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    cols_to_keep = []
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            if X_df[col].std(skipna=True) > 1e-6:
                cols_to_keep.append(col)
        else:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            if X_df[col].std() > 1e-6:
                cols_to_keep.append(col)

    X_df = X_df[cols_to_keep]
    X_df.fillna(X_df.median(), inplace=True)

    feature_names = X_df.columns.tolist()

    if len(X_df) > max_samples:
        print(f"[*] Stratified sampling {max_samples:,} records for high-speed processing...", flush=True)
        X_df, _, y_binary, _ = train_test_split(
            X_df, y_binary, train_size=max_samples, stratify=y_binary, random_state=42
        )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y_binary.values, test_size=0.25, stratify=y_binary.values, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    del df, X_df, X_train_raw, X_test_raw
    gc.collect()

    return X_train_scaled, y_train, X_test_scaled, y_test, feature_names
