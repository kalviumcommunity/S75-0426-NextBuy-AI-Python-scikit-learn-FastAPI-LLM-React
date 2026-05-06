from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return pipeline