# 📊 Understanding Supervised Learning Problem Types

This project demonstrates how to correctly identify and implement different supervised learning problem types.

---

## 🎯 Objective

To understand:

- Classification vs Regression
- Binary, Multi-class, Multi-label problems
- Proper evaluation metrics
- Real-world ML problem framing

---

## 📌 Scenario Analysis

### 1️⃣ Fraud Detection
- Type: Binary Classification
- Target: Fraud (Yes/No)
- Metrics: Precision, Recall, F1, ROC-AUC
- Why: Output is categorical

---

### 2️⃣ House Price Prediction
- Type: Regression
- Target: Price
- Metrics: MAE, RMSE, R²
- Why: Output is continuous

---

### 3️⃣ Movie Genre Tagging
- Type: Multi-label Classification
- Target: Multiple genres
- Metrics: F1-score, Hamming Loss
- Why: Multiple labels per sample

---

### 4️⃣ Sales Prediction
- Type: Regression (Count)
- Target: Units sold
- Metrics: MAE, RMSE
- Why: Output is numeric

---

### 5️⃣ Disease Classification
- Type: Multi-class Classification
- Target: Disease category
- Metrics: Accuracy, F1-score
- Why: Only one class per sample

---

## ❗ Why NOT Other Types?

- Regression for classification → Wrong (no meaning in numeric labels)
- Multi-class instead of multi-label → Loses multiple tags
- Accuracy in fraud detection → Misleading due to imbalance

---

## 🧪 Code Implementation

### ✔️ Classification Example
- Logistic Regression
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### ✔️ Regression Example
- Linear Regression
- Metrics:
  - MAE
  - RMSE
  - R² Score

---

## ▶️ Run Code

```bash
python problem_type_analysis.py
```

---

## 🧠 Key Learnings

- Always identify problem type first
- Choose correct metric based on problem
- Classification ≠ Regression
- Multi-label ≠ Multi-class

---

## 🏁 Summary

✔️ Problem type defines everything  
✔️ Metrics must match the problem  
✔️ Wrong framing = wrong model  

---

> This project focuses on understanding ML problem types, not model accuracy.