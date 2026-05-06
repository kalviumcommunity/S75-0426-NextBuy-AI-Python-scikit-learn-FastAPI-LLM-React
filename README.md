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
project-root/
│
├── data/
│ └── sample.csv
│
├── models/ # Saved artifacts (ignored in git)
│
├── reports/
│
├── src/
│ ├── init.py
│ ├── config.py
│ ├── data_loader.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── requirements.txt
└── README.md


---

## ⚙️ How It Works

### 🔹 Training Pipeline (`src/train.py`)

1. Load raw data using `data_loader.py`
2. Split data into train and test sets
3. Fit preprocessing pipeline on **training data only**
4. Train model
5. Evaluate on test data
6. Save artifacts (`model.pkl`, `pipeline.pkl`)

👉 `.fit()` happens ONLY here

---

### 🔹 Inference Pipeline (`src/predict.py`)

1. Load saved model + pipeline
2. Apply `.transform()` (NOT `.fit_transform()`)
3. Generate predictions

👉 No training happens here

---

## ▶️ How to Run

```bash
python -m src.train
python -m src.predict