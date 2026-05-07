# src/train.py

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

from src.persistence import save_model


# =========================
# LOAD DATASET
# =========================

# Change filename if needed
df = pd.read_csv("data/raw/house_price.csv")


# =========================
# FEATURES AND TARGET
# =========================

# Change target column if needed
TARGET_COLUMN = "price"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# BASELINE MODEL
# =========================

baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)

baseline_preds = baseline.predict(X_test)


# =========================
# LINEAR REGRESSION MODEL
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

model_preds = model.predict(X_test)


# =========================
# SAVE MODEL
# =========================

save_model(model, "models/linear_regression.pkl")


# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    print("\n==============================")
    print(name)
    print("==============================")

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")


# =========================
# PRINT RESULTS
# =========================

evaluate_model(
    "Baseline Model",
    y_test,
    baseline_preds
)

evaluate_model(
    "Linear Regression Model",
    y_test,
    model_preds
)


# =========================
# CROSS VALIDATION
# =========================

cv_r2 = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

cv_mse = -cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

cv_rmse = np.sqrt(cv_mse)

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")

print(f"R² Scores     : {cv_r2}")
print(f"Mean R²       : {cv_r2.mean():.3f}")
print(f"Std R²        : {cv_r2.std():.3f}")

print()

print(f"RMSE Scores   : {cv_rmse}")
print(f"Mean RMSE     : {cv_rmse.mean():.3f}")
print(f"Std RMSE      : {cv_rmse.std():.3f}")


# =========================
# COEFFICIENTS
# =========================

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n==============================")
print("MODEL COEFFICIENTS")
print("==============================")

print(f"Intercept : {model.intercept_}")

print()

print(coef_df)