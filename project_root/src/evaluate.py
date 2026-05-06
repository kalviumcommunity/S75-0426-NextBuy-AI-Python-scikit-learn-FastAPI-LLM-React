from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute evaluation metrics and return as a dictionary."""
    predictions = model.predict(X_test)
    return {
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1': f1_score(y_test, predictions, average='weighted')
    }