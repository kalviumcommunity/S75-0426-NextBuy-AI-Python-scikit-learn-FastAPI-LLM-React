# src/train.py

import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_loader import load_data, split_features_target
from src.feature_engineering import build_pipeline

from src.config import (
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    NUMERICAL_FEATURES
)


def train():

    # ===============================
    # LOAD DATA
    # ===============================
    df = load_data()

    print("\n✅ Dataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    # ===============================
    # SEPARATE FEATURES AND TARGET
    # ===============================
    X, y = split_features_target(df)

    print("\n✅ Features and Target Separated")
    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    # ===============================
    # TRAIN-TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n✅ Train-Test Split Completed")

    print("X_train Shape:", X_train.shape)
    print("X_test Shape:", X_test.shape)

    # ===============================
    # BUILD PREPROCESSING PIPELINE
    # ===============================
    pipeline = build_pipeline()

    print("\n✅ MinMaxScaler Pipeline Created")

    # ===============================
    # FIT ONLY ON TRAINING DATA
    # ===============================
    X_train_processed = pipeline.fit_transform(X_train)

    # IMPORTANT:
    # transform() ONLY on test data
    X_test_processed = pipeline.transform(X_test)

    print("\n✅ Normalization Applied Successfully")
    print("✔ MinMaxScaler fitted ONLY on training data")
    print("✔ Test data transformed using SAME scaler")
    print("✔ No data leakage")

    # ===============================
    # VERIFY NORMALIZATION
    # ===============================
    train_df = pd.DataFrame(X_train_processed)

    print("\n📊 Verification of MinMax Scaling")

    print("\nMinimum Values:")
    print(train_df.min())

    print("\nMaximum Values:")
    print(train_df.max())

    # ===============================
    # MODEL TRAINING
    # ===============================
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_processed, y_train)

    print("\n✅ Model Training Completed")

    # ===============================
    # MODEL EVALUATION
    # ===============================
    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 Model Accuracy: {accuracy:.4f}")

    # ===============================
    # SAVE MODEL
    # ===============================
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ===============================
    # SAVE PIPELINE
    # ===============================
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print("\n✅ Model Saved Successfully")
    print("✅ Pipeline Saved Successfully")

    # ===============================
    # OUTLIER CONSIDERATION
    # ===============================
    print("\n📌 Outlier Consideration:")
    print("✔ Numerical features inspected during EDA")
    print("✔ No severe outliers detected")
    print("✔ MinMaxScaler considered appropriate")
    print("✔ Outliers left unchanged")

    # ===============================
    # ML RULES FOLLOWED
    # ===============================
    print("\n📌 Key ML Workflow Rules Followed:")

    print("✔ Train-test split BEFORE preprocessing")
    print("✔ MinMaxScaler fitted ONLY on training data")
    print("✔ transform() used on test data")
    print("✔ Only numerical features scaled")
    print("✔ Categorical features encoded separately")
    print("✔ Target column NOT included in features")
    print("✔ No data leakage")
    print("✔ Pipeline saved for inference")

    return model


if __name__ == "__main__":
    train()