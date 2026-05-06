# src/train.py

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, split_features_target
from src.feature_engineering import build_pipeline
from src.config import MODEL_PATH, PIPELINE_PATH, RANDOM_STATE


def train():
    df = load_data()

    # ✅ Proper separation
    X, y = split_features_target(df)

    # ✅ Train-test split BEFORE fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    # ✅ Fit ONLY on training data
    X_train_processed = pipeline.fit_transform(X_train)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save pipeline
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print("\n✅ Training completed and artifacts saved!")

    return model


if __name__ == "__main__":
    train()