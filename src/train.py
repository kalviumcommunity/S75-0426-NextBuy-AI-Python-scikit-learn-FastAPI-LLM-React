import pickle
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.feature_engineering import build_pipeline
from src.config import DATA_PATH, MODEL_PATH, PIPELINE_PATH, RANDOM_STATE


def train():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save pipeline
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return model, pipeline, X_test, y_test


if __name__ == "__main__":
    train()
    print("✅ Training completed and artifacts saved!")