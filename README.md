# 🛒 NextBuy-AI

NextBuy-AI is a machine learning project focused on:

- Leakage-free preprocessing
- Proper normalization using MinMaxScaler
- Clean ML pipeline architecture
- Reproducible training workflows
- Production-ready preprocessing pipelines

The project demonstrates how to correctly normalize numerical features while preventing train-test contamination.

---

# 🚀 Features

- 🎯 Explicit feature and target definition
- 🔢 Numerical feature normalization using MinMaxScaler
- 🏷️ Separate handling of categorical features
- 🔒 Leakage prevention
- ⚖️ Proper train-test splitting
- 💾 Saved preprocessing pipeline
- 📊 Scaling verification
- ⚡ Production-ready prediction workflow

---

# 🧱 Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib

---

# 📂 Project Structure

```text
project-root/
│
├── data/
│   └── sample.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# 🎯 Feature Configuration

## Target Column

```python
TARGET_COLUMN = "target"
```

---

## Numerical Features

```python
NUMERICAL_FEATURES = [
    "feature1",
    "feature2"
]
```

These features are normalized using MinMaxScaler.

---

## Categorical Features

```python
CATEGORICAL_FEATURES = [
    "category1"
]
```

Categorical features are encoded separately using OneHotEncoder.

---

# 🔢 Why MinMaxScaler Was Chosen

MinMaxScaler was selected because:

- The model is sensitive to feature magnitudes
- Features exist on different numerical scales
- Normalization keeps all values within a fixed range
- It improves optimization stability

MinMaxScaler transforms features using:

```python
x_scaled = (x - x_min) / (x_max - x_min)
```

After transformation:

- Minimum value becomes 0
- Maximum value becomes 1

---

# ⚖️ Proper Train-Test Splitting

The dataset is split BEFORE preprocessing.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This prevents test-set information from leaking into training.

---

# 🔒 Leakage Prevention

The following precautions are implemented:

- Train-test split BEFORE scaling
- MinMaxScaler fitted ONLY on training data
- Test data transformed using the SAME scaler
- Target column excluded from preprocessing
- Only numerical features scaled
- Categorical features handled separately
- transform() used during prediction

Incorrect approach:

```python
scaler.fit_transform(X)
train_test_split(X, y)
```

Correct approach:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler.fit(X_train)
scaler.transform(X_test)
```

---

# 🏗️ Preprocessing Pipeline

The project uses:

- MinMaxScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer for clean preprocessing

```python
ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(), CATEGORICAL_FEATURES)
    ]
)
```

---

# 📊 Scaling Verification

After normalization:

- Minimum training values ≈ 0
- Maximum training values ≈ 1

Verification logs are printed during training.

---

# 💾 Artifact Persistence

The trained model and preprocessing pipeline are saved using pickle.

```python
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(pipeline, open(PIPELINE_PATH, "wb"))
```

---

# ⚡ Prediction Workflow

During inference:

- Saved pipeline is loaded
- ONLY transform() is used
- fit() or fit_transform() is NEVER used

```python
processed = pipeline.transform(input_df)
predictions = model.predict(processed)
```

This guarantees preprocessing consistency between training and prediction.

---

# ⚠️ Outlier Consideration

Numerical feature distributions were inspected during EDA.

Observations:

- No severe outliers detected
- MinMaxScaler considered acceptable
- Features left unchanged

If severe outliers appear in future datasets:

- RobustScaler
- Log transformation
- Outlier capping

may be considered.

---

# ▶️ Run Training

```bash
python -m src.train
```

---

# ▶️ Run Prediction

```bash
python -m src.predict
```

---

# 🏁 Summary

This project demonstrates:

- Proper normalization using MinMaxScaler
- Leakage-free preprocessing
- Correct train-test boundaries
- Clean ML pipeline architecture
- Reproducible preprocessing
- Production-ready prediction workflow

---

Built for disciplined ML engineering and preprocessing best practices.