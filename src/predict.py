# src/predict.py

import pickle
import pandas as pd

from src.config import (
    MODEL_PATH,
    PIPELINE_PATH
)


def load_artifacts():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    return model, pipeline


def predict(input_df: pd.DataFrame):

    # ===============================
    # LOAD ARTIFACTS
    # ===============================
    model, pipeline = load_artifacts()

    print("\n✅ Model and Pipeline Loaded")

    # ===============================
    # IMPORTANT:
    # ONLY transform()
    # NEVER fit_transform()
    # ===============================
    processed_data = pipeline.transform(input_df)

    predictions = model.predict(processed_data)

    return predictions


if __name__ == "__main__":

    sample = pd.read_csv("data/sample.csv")

    preds = predict(sample)

    print("\n📊 Predictions:")
    print(preds)