# problem_type_analysis.py

"""
This script demonstrates:
1. Problem type identification for real-world scenarios
2. Simple classification example
3. Simple regression example
"""

import pandas as pd
import numpy as np

# Classification imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Regression imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 🔹 PART 1: SCENARIO ANALYSIS
# =====================================================

def print_scenarios():
    print("\n===== SCENARIO ANALYSIS =====\n")

    scenarios = [
        {
            "name": "Fraud Detection",
            "type": "Binary Classification",
            "target": "Fraud (Yes/No)",
            "metrics": "Precision, Recall, F1-score, ROC-AUC",
            "why": "Output is categorical (fraud or not)"
        },
        {
            "name": "House Price Prediction",
            "type": "Regression",
            "target": "Price (continuous value)",
            "metrics": "MAE, RMSE, R2",
            "why": "Output is a numerical value"
        },
        {
            "name": "Movie Genre Tagging",
            "type": "Multi-label Classification",
            "target": "Multiple genres",
            "metrics": "F1-score, Hamming Loss",
            "why": "Multiple labels can exist simultaneously"
        },
        {
            "name": "Sales Prediction",
            "type": "Regression (Count)",
            "target": "Number of units sold",
            "metrics": "MAE, RMSE",
            "why": "Output is a number"
        },
        {
            "name": "Disease Classification",
            "type": "Multi-class Classification",
            "target": "Disease category (Viral/Bacterial/Autoimmune)",
            "metrics": "Accuracy, F1-score",
            "why": "Only one category per patient"
        }
    ]

    for s in scenarios:
        print(f"📌 {s['name']}")
        print(f"   Problem Type: {s['type']}")
        print(f"   Target: {s['target']}")
        print(f"   Metrics: {s['metrics']}")
        print(f"   Reason: {s['why']}\n")


# =====================================================
# 🔹 PART 2: CLASSIFICATION EXAMPLE
# =====================================================

def classification_example():
    print("\n===== CLASSIFICATION EXAMPLE =====\n")

    # Synthetic data
    X = pd.DataFrame({
        "feature1": np.random.randint(1, 100, 100),
        "feature2": np.random.randint(1, 50, 100)
    })

    # Binary target (0 or 1)
    y = (X["feature1"] + X["feature2"] > 75).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))


# =====================================================
# 🔹 PART 3: REGRESSION EXAMPLE
# =====================================================

def regression_example():
    print("\n===== REGRESSION EXAMPLE =====\n")

    # Synthetic data
    X = pd.DataFrame({
        "size": np.random.randint(500, 3000, 100),
        "bedrooms": np.random.randint(1, 5, 100)
    })

    # Continuous target
    y = X["size"] * 50 + X["bedrooms"] * 10000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2 Score:", r2_score(y_test, preds))


# =====================================================
# 🔹 MAIN
# =====================================================

if __name__ == "__main__":
    print_scenarios()
    classification_example()
    regression_example()