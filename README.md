# 🏚️ Predicting Structural Damage with Logistic Regression

This project applies logistic regression to predict the likelihood of **severe structural damage** in buildings affected by an earthquake in Nepal, using data provided in an SQLite database.

---

## 📌 Project Summary

The goal of this project is to build a **binary classification model** that predicts whether a building experienced **severe damage (damage grade > 3)** using pre-earthquake structural features. This helps assess risk and inform rebuilding priorities.

---

## 🔍 Data Description

- **Source**: A relational SQLite database (`nepal.sqlite`)
- **Tables used**:
  - `id_map`
  - `building_structure`
  - `building_damage`
- **Filtered for**: `district_id = 4`
- **Target Variable**: `severe_damage` (1 if damage grade > 3, else 0)
- **Features used**: Structural attributes of the building (e.g., foundation type, roof type, floor count)


---

## 🧠 Methodology

1. **Data Wrangling**:
   - Merged 3 related tables using SQL
   - Removed post-earthquake and irrelevant columns
   - Converted `damage_grade` to binary `severe_damage`

2. **Modeling**:
   - Train/test split (80/20)
   - Categorical encoding with `category_encoders.OneHotEncoder`
   - Logistic Regression with `max_iter=1000`

3. **Evaluation**:
   - **Baseline Accuracy**: 0.64 (predicting majority class only)
   - **Training Accuracy**: 0.71
   - **Test Accuracy**: 0.72

---

## 📦 Libraries Used

- Python (pandas, numpy, seaborn, matplotlib)
- scikit-learn
- category_encoders
- sqlite3

---

## 📈 Results

| Metric             | Value  |
|--------------------|--------|
| Baseline Accuracy  | 0.64   |
| Training Accuracy  | 0.71   |
| Test Accuracy      | 0.72   |

*The model improves upon the baseline by ~8%, demonstrating meaningful predictive power based on structural features.*

---


