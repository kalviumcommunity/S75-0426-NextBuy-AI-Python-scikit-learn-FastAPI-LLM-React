# 🛒 NextBuy-AI

**NextBuy-AI** is a machine learning project built to demonstrate:

- Clean ML architecture
- Proper feature and target definition
- Feature type grouping
- Feature distribution analysis
- Leakage-free train-test splitting
- Reproducible ML workflows

The goal is not only to train a model, but to build a **production-ready and trustworthy ML pipeline**.

---

# 🚀 Features

- 🎯 Explicit target definition
- 🔢 Manual numerical and categorical feature grouping
- 🔒 Leakage prevention
- 📊 Exploratory Data Analysis (EDA)
- ⚖️ Proper train-test splitting
- 💾 Artifact-based workflow
- ⚡ Independent training and inference pipelines

---

# 🧱 Tech Stack

- Python
- scikit-learn
- pandas
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
├── reports/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── eda.py
│   └── leakage_demo.py
│
├── requirements.txt
└── README.md
```

---

# 🎯 Feature and Target Definition

## Target Variable

| Property | Description |
|---|---|
| Target Column | `target` |
| Problem Type | Classification |
| Business Meaning | Predicts the final outcome based on input features |

---

# 🔢 Numerical Features

| Feature | Reason |
|---|---|
| feature1 | Represents measurable numeric values |
| feature2 | Continuous numeric quantity |

## Scaling Strategy

Numerical features may be scaled using `StandardScaler` during preprocessing.

---

# 🏷️ Categorical Features

| Feature | Type | Reason |
|---|---|---|
| category1 | Nominal | Represents category labels without order |

## Encoding Strategy

Categorical features may use:
- One-Hot Encoding
- Ordinal Encoding (if ordered categories exist)

---

# ❌ Excluded Columns

| Column | Reason Excluded |
|---|---|
| id | Unique identifier with no predictive value |
| timestamp | Not available during real-world prediction |

---

# ⚠️ Edge Case Handling

## Binary Features

Binary columns stored as `0/1` are treated conceptually based on meaning, not storage type.

## Ordinal Features

Ordinal features are handled separately if category ordering matters.

## High Cardinality Features

High-cardinality categorical columns may require grouping or advanced encoding.

## Timestamp Features

Timestamp columns are excluded unless converted into meaningful features.

---

# 📊 Feature Distribution Analysis

EDA is performed before modeling to understand feature behavior.

## Numerical Inspection

- Summary statistics
- Histograms
- Boxplots
- Skewness analysis

## Categorical Inspection

- Value counts
- Rare category detection
- Imbalance analysis

## Key Goals

- Detect skewness
- Detect outliers
- Detect imbalance
- Identify preprocessing requirements

---

# ⚖️ Data Splitting Strategy

## Split Configuration

| Setting | Value |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Stratification | Enabled |

---

## Why This Strategy?

- Training data is used for learning
- Testing data simulates unseen real-world data
- Stratification preserves class balance

---

# 🔒 Leakage Prevention

The following precautions are implemented:

- Target column excluded from features
- ID columns excluded
- No preprocessing before splitting
- No scaling before splitting
- No encoding before splitting
- Test data remains untouched

---

# 🚨 Data Leakage Demonstration

This project also demonstrates train-test contamination leakage.

## Incorrect Workflow

```python
scaler.fit_transform(X)
train_test_split(X_scaled, y)
```

This leaks test-set information into training.

---

## Correct Workflow

```python
train_test_split(X, y)

scaler.fit(X_train)
scaler.transform(X_test)
```

This preserves honest evaluation.

---

# ⚙️ ML Workflow

## EDA Pipeline

```bash
python -m src.eda
```

Performs:
- Distribution analysis
- Outlier detection
- Skewness inspection

---

## Data Preprocessing

```bash
python -m src.data_preprocessing
```

Performs:
- Validation
- Feature separation
- Train-test split

---

## Training Pipeline

```bash
python -m src.train
```

Performs:
- Model training
- Evaluation
- Artifact saving

---

## Prediction Pipeline

```bash
python -m src.predict
```

Performs:
- Artifact loading
- Prediction generation

---

# 🏁 Summary

This project demonstrates:

- Explicit feature definition
- Correct feature typing
- Data inspection before modeling
- Leakage-free train-test splitting
- Honest ML evaluation
- Production-ready ML workflow design

---

> Built for ML system design learning with emphasis on correctness, reproducibility, and disciplined ML engineering.



## Setup Instructions

### Create Virtual Environment
python -m venv .venv

### Activate Environment
.\.venv\Scripts\Activate

### Install Dependencies
pip install -r requirements.txt

### Run Project
python main.py

# Baseline Model

This project uses a baseline model using DummyClassifier with the strategy "most_frequent".

The baseline model is compared against Logistic Regression using:

- Accuracy
- F1 Score
- Classification Report

Purpose:
- Validate that the ML model performs better than naive predictions.
- Measure actual improvement over a trivial baseline.


# Linear Regression ML Project

## Steps Performed

- Created baseline model
- Trained Linear Regression model
- Compared performance metrics
- Evaluated using:
  - MSE
  - RMSE
  - MAE
  - R2 Score

## Run Project

```bash
python src/train.py



---

# Step 12 — Git Commands

Now save everything.

---

## Check status

```powershell
git status