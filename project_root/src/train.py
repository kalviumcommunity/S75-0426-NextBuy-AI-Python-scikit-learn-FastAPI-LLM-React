from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Load dataset
data = load_iris()

X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# BASELINE MODEL
# =========================

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_preds = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_preds)

baseline_f1 = f1_score(
    y_test,
    baseline_preds,
    average="weighted"
)

# =========================
# REAL MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

model_preds = model.predict(X_test)

model_accuracy = accuracy_score(y_test, model_preds)

model_f1 = f1_score(
    y_test,
    model_preds,
    average="weighted"
)

# =========================
# RESULTS
# =========================

print("\n===== BASELINE MODEL =====")
print("Accuracy:", baseline_accuracy)
print("F1 Score:", baseline_f1)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", model_accuracy)
print("F1 Score:", model_f1)

print("\n===== IMPROVEMENT =====")
print("Accuracy Improvement:", model_accuracy - baseline_accuracy)
print("F1 Improvement:", model_f1 - baseline_f1)

print("\n===== BASELINE REPORT =====")
print(classification_report(y_test, baseline_preds))

print("\n===== MODEL REPORT =====")
print(classification_report(y_test, model_preds))