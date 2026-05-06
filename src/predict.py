import pickle
import pandas as pd
from src.config import MODEL_PATH, PIPELINE_PATH

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    return model, pipeline


def predict(input_df: pd.DataFrame):
    model, pipeline = load_artifacts()

    # ✅ Ensure SAME columns as training
    expected_cols = pipeline.feature_names_in_

    input_df = input_df[expected_cols]

    processed = pipeline.transform(input_df)   # ❗ ONLY transform
    preds = model.predict(processed)

    return preds


if __name__ == "__main__":
    sample = pd.read_csv("data/sample.csv")
    preds = predict(sample)
    print("Predictions:", preds)