# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

from src.data_loader import load_data


def inspect_data():
    # Load data
    df = load_data("data/sample.csv")

    print("\n📊 DATASET SHAPE")
    print(df.shape)

    print("\n📌 DATA TYPES")
    print(df.dtypes)

    # =========================
    # NUMERICAL FEATURES
    # =========================
    print("\n🔢 NUMERICAL FEATURE ANALYSIS")

    for col in NUMERICAL_FEATURES:
        print(f"\n--- {col} ---")

        # Summary stats
        print(df[col].describe())

        # Skewness
        print("Skewness:", df[col].skew())

        # Histogram
        df[col].hist(bins=30)
        plt.title(f"{col} Distribution")
        plt.show()

        # Boxplot
        df.boxplot(column=col)
        plt.title(f"{col} Boxplot")
        plt.show()

    # =========================
    # CATEGORICAL FEATURES
    # =========================
    print("\n🏷️ CATEGORICAL FEATURE ANALYSIS")

    for col in CATEGORICAL_FEATURES:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())
        print("\nNormalized:")
        print(df[col].value_counts(normalize=True))

    # =========================
    # TARGET BASED COMPARISON
    # =========================
    print("\n🎯 TARGET BASED ANALYSIS")

    for col in NUMERICAL_FEATURES:
        print(f"\n--- {col} vs {TARGET_COLUMN} ---")
        print(df.groupby(TARGET_COLUMN)[col].describe())

        df.boxplot(column=col, by=TARGET_COLUMN)
        plt.title(f"{col} vs {TARGET_COLUMN}")
        plt.suptitle("")  # remove default title
        plt.show()


if __name__ == "__main__":
    inspect_data()