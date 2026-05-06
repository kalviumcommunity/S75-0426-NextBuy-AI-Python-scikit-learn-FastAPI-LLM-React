import joblib

def save_artifacts(model, pipeline, model_path: str, pipeline_path: str):
    """Serialize and save model and pipeline."""
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)

def predict(new_data, model, pipeline):
    """Generate predictions using saved artifacts."""
    processed_data = pipeline.transform(new_data)
    return model.predict(processed_data)