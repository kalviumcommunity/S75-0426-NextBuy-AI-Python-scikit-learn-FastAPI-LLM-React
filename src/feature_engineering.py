# src/feature_engineering.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return pipeline