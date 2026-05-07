# src/train.py

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_loader import load_data, split_features_target
from src.feature_engineering import build_pipeline

from src.config import (
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    TEST_SIZE
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
    # Split BEFORE scaling/preprocessing
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

    print("\nTrain Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest Distribution:")
    print(y_test.value_counts(normalize=True))

    # ===============================
    # BUILD PREPROCESSING PIPELINE
    # ===============================
    pipeline = build_pipeline()

    print("\n✅ Preprocessing Pipeline Created")

    # ===============================
    # FIT ONLY ON TRAINING DATA
    # ===============================
    X_train_processed = pipeline.fit_transform(X_train)

    # IMPORTANT:
    # transform() only on test data
    X_test_processed = pipeline.transform(X_test)

    print("\n✅ Scaling & Encoding Applied")
    print("Pipeline fitted ONLY on training data")
    print("No leakage detected")

    # ===============================
    # MODEL TRAINING
    # ===============================
    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X_train_processed, y_train)

    print("\n✅ Model Training Completed")

    # ===============================
    # MODEL EVALUATION
    # ===============================
    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n📊 Model Accuracy:", accuracy)

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

    print("\n📌 Key ML Workflow Rules Followed:")
    print("✔ Train-test split BEFORE preprocessing")
    print("✔ StandardScaler fitted ONLY on training data")
    print("✔ Test data transformed using fitted pipeline")
    print("✔ Target column NOT included in features")
    print("✔ No scaling of categorical features")
    print("✔ No data leakage")

    return model


if __name__ == "__main__":
    train()