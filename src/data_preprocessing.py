# src/data_preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)


# ===============================
# CLEAN DATA
# ===============================
def clean_data(df):
    df = df.dropna()
    return df


# ===============================
# SPLIT DATA
# ===============================
def split_data(df):

    # Validation Checks
    assert TARGET_COLUMN in df.columns
    assert TARGET_COLUMN not in ALL_FEATURES

    # Separate X and y
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print("Numerical Features:", len(NUMERICAL_FEATURES))
    print("Categorical Features:", len(CATEGORICAL_FEATURES))
    print("Total Features:", len(ALL_FEATURES))

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    print("\nTrain Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest Distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


# ===============================
# PREPROCESSOR
# ===============================
def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor