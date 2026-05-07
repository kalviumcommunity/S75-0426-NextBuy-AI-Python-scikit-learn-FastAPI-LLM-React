import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    ALL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)


# ==========================================
# 🧹 DATA CLEANING
# ==========================================
def clean_data(df):
    """
    Simple cleaning step.
    Removes missing values.
    """
    df = df.dropna()
    return df


# ==========================================
# ✂️ TRAIN-TEST SPLIT
# ==========================================
def split_data(df):

    # ------------------------------
    # Validate Target Column
    # ------------------------------
    assert TARGET_COLUMN in df.columns

    # ------------------------------
    # Validate Excluded Columns
    # ------------------------------
    for col in EXCLUDED_COLUMNS:
        assert col not in ALL_FEATURES

    # ------------------------------
    # Separate Features and Target
    # ------------------------------
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # ------------------------------
    # Validation Checks
    # ------------------------------
    print("Numerical features:", len(NUMERICAL_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))
    print("Total features:", len(ALL_FEATURES))

    print("\nNumerical Feature List:")
    print(NUMERICAL_FEATURES)

    print("\nCategorical Feature List:")
    print(CATEGORICAL_FEATURES)

    print("\nExcluded Columns:")
    print(EXCLUDED_COLUMNS)

    print("\nTarget Column:")
    print(TARGET_COLUMN)

    print("\nX Shape:", X.shape)
    print("y Shape:", y.shape)

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ------------------------------
    # Split Verification
    # ------------------------------
    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    print("\nTrain Target Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest Target Distribution:")
    print(y_test.value_counts(normalize=True))

    # ------------------------------
    # Leakage Prevention Reminder
    # ------------------------------
    print("\n✅ No preprocessing fitted before splitting")
    print("✅ Test set remains untouched")

    return X_train, X_test, y_train, y_test