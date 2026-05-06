import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import TARGET_COLUMN, ALL_FEATURES

# Load data
df = pd.read_csv("data/sample.csv")

X = df[ALL_FEATURES]
y = df[TARGET_COLUMN]

print("========== WRONG APPROACH (WITH LEAKAGE) ==========")

# ❌ WRONG: Scaling BEFORE splitting (LEAKAGE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 🚨 uses FULL DATA

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

wrong_acc = accuracy_score(y_test, y_pred)
print("Accuracy (WRONG - Leakage):", wrong_acc)


print("\n========== CORRECT APPROACH (NO LEAKAGE) ==========")

# ✅ CORRECT: Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler ONLY on training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

correct_acc = accuracy_score(y_test, y_pred)
print("Accuracy (CORRECT):", correct_acc)


print("\n========== COMPARISON ==========")
print("Leakage Accuracy:", wrong_acc)
print("Correct Accuracy:", correct_acc)