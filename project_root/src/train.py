from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from src.data_preprocessing import load_and_split_data
from src.persistence import save_model

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()

# -------------------------
# Baseline Model
# -------------------------

baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

# -------------------------
# Linear Regression Model
# -------------------------

model = LinearRegression()

model.fit(X_train, y_train)

model_predictions = model.predict(X_test)
save_model(model, "models/linear_regression.pkl")

# -------------------------
# Evaluation Function
# -------------------------

def evaluate_model(name, y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R2   : {r2:.2f}")

# -------------------------
# Results
# -------------------------

evaluate_model(
    "Baseline Model",
    y_test,
    baseline_predictions
)

evaluate_model(
    "Linear Regression Model",
    y_test,
    model_predictions
)