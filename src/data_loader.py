# src/data_loader.py

import pandas as pd
from src.config import TARGET_COLUMN, ALL_FEATURES


def load_data(filepath="data/sample.csv"):
    df = pd.read_csv(filepath)

    # ===============================
    # ✅ VALIDATIONS
    # ===============================
    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    for col in ALL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' missing in dataset")

    if TARGET_COLUMN in ALL_FEATURES:
        raise ValueError("Target column is included in features (DATA LEAKAGE!)")

    return df


def split_features_target(df):
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print("\n✅ Feature Shape:", X.shape)
    print("✅ Target Shape:", y.shape)

    print("\n📊 Target Distribution:")
    print(y.value_counts(normalize=True))

    return X, y