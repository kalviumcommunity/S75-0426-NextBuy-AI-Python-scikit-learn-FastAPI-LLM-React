import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    "Size": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5, 5],
    "Price": [40, 50, 60, 75, 90, 100, 115, 130]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Size", "Bedrooms"]]
y = df["Price"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Baseline Model
# -----------------------------
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_preds = baseline.predict(X_test)

# -----------------------------
# Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

model_preds = model.predict(X_test)

# -----------------------------
# MAE Evaluation
# -----------------------------
baseline_mae = mean_absolute_error(y_test, baseline_preds)
model_mae = mean_absolute_error(y_test, model_preds)

# Other Metrics
rmse = np.sqrt(mean_squared_error(y_test, model_preds))
r2 = r2_score(y_test, model_preds)

# -----------------------------
# Results
# -----------------------------
print("\n===== MODEL EVALUATION =====")

print(f"\nBaseline MAE: {baseline_mae:.2f}")
print(f"Linear Regression MAE: {model_mae:.2f}")

print(f"\nRMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

improvement = baseline_mae - model_mae

print(f"\nImprovement over baseline: {improvement:.2f}")

# -----------------------------
# Cross Validation
# -----------------------------
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

mae_scores = -cv_scores

print("\nCross Validation MAE Scores:")
print(mae_scores)

print(f"\nAverage CV MAE: {mae_scores.mean():.2f}")