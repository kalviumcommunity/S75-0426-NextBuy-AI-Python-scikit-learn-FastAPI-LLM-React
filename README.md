# 🛒 NextBuy-AI

**NextBuy-AI** is a machine learning project built to demonstrate **clean architecture, proper separation of concerns, correct feature/target definition, and data understanding before modeling**.

This project strictly separates:
- Data Loading
- Model Training
- Inference (Prediction)

It also ensures:
- Clear **target variable definition**
- Proper **feature selection**
- **Zero data leakage**
- **Feature distribution inspection before modeling**

The goal is not just building a model — but building a **reliable, reproducible, and production-ready ML pipeline**.

---

## 🚀 Features

* 🎯 **Separation of Concerns**
  Data loading, training, and inference are completely independent

* 🧠 **Explicit Feature Definition**
  Features are manually defined (no auto-selection)

* 🔒 **No Data Leakage**
  Target is never included in feature set

* 📊 **Feature Distribution Analysis**
  Data is inspected before modeling to detect skewness, outliers, and imbalance

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
project-root/
│
├── data/
│ └── sample.csv
│
├── models/ # Saved artifacts (ignored in git)
│ ├── model.pkl
│ └── pipeline.pkl
│
├── reports/
│
├── src/
│ ├── init.py
│ ├── config.py # Feature + target definitions
│ ├── data_loader.py # ONLY loads raw data
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py # Training logic
│ ├── evaluate.py
│ ├── predict.py # Inference logic
│ └── eda.py # Feature distribution inspection
│
├── requirements.txt
└── README.md

---

## ⚙️ How It Works

### 🔹 Training Pipeline (`src/train.py`)

1. Load raw data  
2. Validate dataset  
3. Split into train/test  
4. Fit preprocessing pipeline on **training data only**  
5. Train model  
6. Evaluate performance  
7. Save:
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

## ▶️ How to Run

```bash
# Run EDA (Feature Inspection)
python -m src.eda

# Train model
python -m src.train

# Run prediction
python -m src.predict