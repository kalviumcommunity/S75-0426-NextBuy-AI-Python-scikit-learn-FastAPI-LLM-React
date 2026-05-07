# src/feature_engineering.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


def build_pipeline():

    # ===============================
    # NUMERICAL TRANSFORMER
    # ===============================
    numerical_transformer = Pipeline([
        ("scaler", MinMaxScaler())
    ])

    # ===============================
    # CATEGORICAL TRANSFORMER
    # ===============================
    categorical_transformer = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ])

    # ===============================
    # COLUMN TRANSFORMER
    # ===============================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor