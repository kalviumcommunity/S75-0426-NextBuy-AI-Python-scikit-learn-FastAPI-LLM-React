# 🛒 NextBuy-AI

**NextBuy-AI** is a machine learning project built to demonstrate **clean architecture and proper separation of concerns** in ML systems.

This project strictly separates:
- Data Loading
- Model Training
- Inference (Prediction)

The goal is not just building a model — but building a **reliable, reusable, and production-ready ML pipeline**.

---

## 🚀 Features

* 🎯 **Clear Separation of Concerns**
  Data loading, training, and inference are implemented independently

* 🧠 **Proper ML Pipeline Design**
  Preprocessing is fitted only during training and reused during prediction

* 💾 **Artifact-Based Workflow**
  Model and preprocessing pipeline are saved and reused

* ⚡ **Independent Execution**
  Training and prediction can run separately without dependency

---

## 🧱 Tech Stack

* **Language:** Python
* **Machine Learning:** scikit-learn
* **Environment:** venv (Virtual Environment)

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
├── reports/                 # Optional evaluation outputs
│
├── src/
│   ├── __init__.py          # Makes src a package
│   ├── config.py            # Paths & constants
│   ├── data_loader.py       # ONLY loads raw data
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py             # Training logic (fit happens here)
│   ├── evaluate.py
│   └── predict.py           # Inference logic (NO fitting)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 🔹 Training Pipeline (`src/train.py`)

1. Load raw data using `data_loader.py`
2. Split data into train and test sets
3. Fit preprocessing pipeline on **training data only**
4. Train model using processed features
5. Evaluate on test data
6. Save:
   - Trained model → `models/model.pkl`
   - Preprocessing pipeline → `models/pipeline.pkl`

👉 **Important:**
- `.fit()` and `.fit_transform()` happen ONLY here
- No inference logic is included

---

### 🔹 Inference Pipeline (`src/predict.py`)

1. Load saved model and pipeline
2. Validate input data
3. Apply `.transform()` (NOT `.fit_transform()`)
4. Generate predictions using `.predict()`

👉 **Important:**
- NO training happens here
- NO preprocessing fitting
- Only uses saved artifacts

---

## ▶️ How to Run

### 1️⃣ Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2️⃣ Run Training

```bash
python -m src.train
```

✔️ This will:
- Train model
- Save artifacts in `models/`

---

### 3️⃣ Run Prediction

```bash
python -m src.predict
```

✔️ This will:
- Load saved model
- Generate predictions on new data

---

## 🧠 Key Design Principles

* ✅ **Separation of Concerns**
  - Data loading ≠ Training ≠ Inference

* ✅ **No Data Leakage**
  - Preprocessing is fitted only on training data

* ✅ **Artifact Reuse**
  - Model and pipeline are saved and reused

* ✅ **Reproducibility**
  - Same pipeline used in both training and prediction

* ✅ **Independent Execution**
  - Training and prediction scripts run separately

---

## ⚠️ Common Mistakes Avoided

❌ Fitting preprocessing during prediction  
❌ Retraining model during inference  
❌ Mixing training and prediction logic  
❌ Using test data during training  
❌ Hardcoding logic in one file  

---

## 🎯 Project Objective

To demonstrate how real-world ML systems should be built using:

- Clean architecture
- Modular design
- Safe training vs inference separation

---

## 🏁 Summary

**Training produces artifacts.**  
**Inference consumes artifacts.**

This project ensures both flows are **completely independent and production-safe**.

---

> Built as part of ML system design learning — focusing on correctness over complexity.