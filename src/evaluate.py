# src/evaluate.py

from sklearn.metrics import accuracy_score

def evaluate(model, pipeline, X_test, y_test):
    X_test_processed = pipeline.transform(X_test)
    predictions = model.predict(X_test_processed)

    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc)

    return acc