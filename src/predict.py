# src/predict.py

import pickle
from src.config import MODEL_PATH, PIPELINE_PATH

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    return model, pipeline

def predict(new_data):
    model, pipeline = load_artifacts()

    processed_data = pipeline.transform(new_data)  # ❗ ONLY transform
    predictions = model.predict(processed_data)

    return predictions