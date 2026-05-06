# 🛒 NextBuy-AI

**NextBuy-AI** is a machine learning project built to demonstrate **clean architecture, proper separation of concerns, correct feature/target definition, data understanding, and leakage-free evaluation**.

This project strictly separates:
- Data Loading
- Model Training
- Inference (Prediction)

It also ensures:
- Clear **target variable definition**
- Proper **feature selection**
- **Zero data leakage**
- **Feature distribution inspection before modeling**
- **Correct train-test split for honest evaluation**

The goal is not just building a model — but building a **reliable, reproducible, and production-ready ML pipeline**.

---

## 🚀 Features

* 🎯 **Separation of Concerns**  
  Data loading, training, and inference are completely independent  

* 🧠 **Explicit Feature Definition**  
  Features are manually defined (no auto-selection)  

* 🔒 **No Data Leakage**  
  Target is never included in feature set  

* 📊 **Feature Distribution Analysis (EDA)**  
  Detects skewness, outliers, and imbalance before modeling  

* ⚖️ **Proper Train-Test Split**  
  Ensures realistic evaluation using unseen data  

* 💾 **Artifact-Based Workflow**  
  Model and preprocessing pipeline are saved and reused  

* ⚡ **Independent Execution**  
  Training and prediction run separately  

---

## 🧱 Tech Stack

* **Language:** Python  
* **Machine Learning:** scikit-learn  
* **Visualization:** matplotlib  
* **Environment:** venv  

---

## 📂 Project Structure

```
project-root/
│
├── data/
│   └── sample.csv
│
├── models/                  # Saved artifacts (ignored in git)
│   ├── model.pkl
│   └── pipeline.pkl
│
├── reports/
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Feature + target definitions
│   ├── data_loader.py       # ONLY loads raw data
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── data_split.py        # Train-test splitting logic
│   ├── train.py             # Training logic
│   ├── evaluate.py
│   ├── predict.py           # Inference logic
│   └── eda.py               # Feature distribution inspection
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 🔹 EDA Pipeline (`src/eda.py`)

1. Load dataset  
2. Analyze numerical features:
   - Summary statistics
   - Skewness
   - Histograms
   - Boxplots  
3. Analyze categorical features:
   - Value counts
   - Imbalance detection  
4. Compare features with target  
5. Identify:
   - Outliers  
   - Skewness  
   - Useful features  

👉 No model training or fitting happens here  

---

### 🔹 Data Splitting (`src/data_split.py`)

1. Load dataset  
2. Validate target and features  
3. Separate:
   - **X (features)**
   - **y (target)**  
4. Perform train-test split:
   - 80% training
   - 20% testing  
5. Apply **stratification (for classification)**  
6. Verify:
   - Shapes
   - Target distribution  

👉 No preprocessing is applied before splitting  
👉 Test set remains completely untouched  

---

### 🔹 Training Pipeline (`src/train.py`)

1. Load raw data  
2. Split into train/test  
3. Fit preprocessing pipeline on **training data only**  
4. Train model  
5. Evaluate performance  
6. Save:
   - `model.pkl`
   - `pipeline.pkl`

👉 `.fit()` and `.fit_transform()` happen ONLY here  

---

### 🔹 Inference Pipeline (`src/predict.py`)

1. Load saved model and pipeline  
2. Validate input data  
3. Apply `.transform()` (NOT `.fit_transform()`)  
4. Generate predictions  

👉 No training happens here  
👉 No preprocessing fitting  

---

## ▶️ How to Run

```bash
# Run EDA (Feature Inspection)
python -m src.eda

# Run Data Split
python -m src.data_split

# Train model
python -m src.train

# Run prediction
python -m src.predict
```

---

## 🎯 Feature and Target Definition

- **Target Column:** Defined explicitly in `config.py`
- **Problem Type:** (e.g., Binary Classification / Regression)

### Feature Categories:
- **Numerical Features:** Explicitly listed
- **Categorical Features:** Explicitly listed

### Excluded Columns:
- IDs, irrelevant columns, or leakage-prone features

### Leakage Prevention:
- Target is NOT included in features
- No future or derived data used

---

## 📊 Feature Distribution Analysis

### Numerical Features:
- Skewness checked using `.skew()`
- Histograms used to inspect distribution
- Boxplots used to detect outliers  

### Categorical Features:
- Value counts analyzed
- Rare categories identified
- Imbalance detected  

### Key Observations:
- Outliers identified in high-value ranges
- Some features show skewness (may require transformation)
- Certain categories dominate distribution  

### Conclusion:
- Data inspection performed **before modeling**
- Transformations will be applied based on evidence  

---

## 📊 Data Splitting Strategy

- Split Ratio: **80% Training / 20% Testing**
- Random State: **42 (for reproducibility)**
- Stratification: **Used for classification problems**

### Why This Strategy?

- Training data is used to learn patterns  
- Testing data simulates **real-world unseen data**  
- Prevents model from memorizing  

### Leakage Prevention:

- No scaling before splitting  
- No encoding before splitting  
- No resampling before splitting  

### Verification:

- Checked shapes of training and testing sets  
- Verified class distribution consistency  
- Ensured reproducibility  

### Key Principle:

**Training learns. Testing evaluates. They must remain separate.**

---

## ⚠️ Common Mistakes Avoided

❌ Using target inside features  
❌ Performing preprocessing before splitting  
❌ Training on test data  
❌ Ignoring class imbalance  
❌ Skipping data inspection  
❌ Mixing training and inference logic  

---

## 🏁 Summary

- Data is **understood before modeling**
- Features and target are **explicitly defined**
- Data is **split correctly before training**
- Model training and inference are **fully separated**

👉 **Training produces artifacts**  
👉 **Inference consumes artifacts**  

This ensures a **clean, reproducible, and production-ready ML system**

---

> Built as part of ML system design learning — focusing on correctness, not shortcuts.