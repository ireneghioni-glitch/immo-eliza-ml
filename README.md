# Immo-Eliza Machine Learning Model 🚀

# Immo Eliza – Real Estate Price Prediction Pipeline
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Sprint](https://img.shields.io/badge/Sprint-3%20Completed-brightgreen)
![Status](https://img.shields.io/badge/Status-ML%20Pipeline%20working-brightgreen)
![Next](https://img.shields.io/badge/Next-Sprint%204%20🏁-lightgrey)
[![BeCode](https://img.shields.io/badge/Training-BeCode-brilliantgreen?logo=becode&logoColor=white)](https://becode.org/)


**Repository:** `immo-eliza-ml`
**Type:** `Learning + Consolidation`
**Duration:** `5 days`
**Deadline:** `03/07/2026, 4:30 pm`
**Team:** solo

---

## 📊 Project Overview
This repository contains an end-to-end Machine Learning pipeline tailored to predict real estate property prices in Belgium using an optimized **XGBoost Regressor**, achieving a validated **R² score of 77.14%**.  
This was the Sprint 3 of 4 of ImmoEliza project.

---

## ⭐️ Project Summary (STAR Method)

### 📌 Situation
The objective was to deploy a predictive Machine Learning model for Immo Eliza based on highly volatile and noisy scraped real estate data from the Belgian market, which featured heavy price asymmetry, missing fields, and significant structural anomalies.

### 🎯 Task
Develop a robust feature preprocessing pipeline to handle missing values, optimize categorical variable encodings, and isolate experimental modeling from final test validation. The target was to surpass linear baselines and achieve an R² performance metric within the 70-80% threshold specified by the coaching team.

### 🛠️ Action
1. **Pipeline Isolation:** Built a strict initial data-splitting step to safeguard a 20% validation/test set, completely preventing data leakage across the entire development lifecycle.
2. **Advanced Preprocessing:** Engineered explicit handling strategies for missing feature values and mapped categorical dimensions using optimized numerical encodings.
3. **Outlier Mitigation Strategy:** Identified and stripped extreme price points (the 1% distribution tails) exclusively within the training matrix (`X_train_clean`), stabilizing the gradient boosting residual weight function.
4. **Hyperparameter Tuning:** Conducted a comprehensive multi-stage grid search (`GridSearchCV`) tracking the $R^2$ coefficient, expanding execution boundaries to a regularized design space (`max_depth=5`, `n_estimators=600`, `min_child_weight=6`).
5. **Production Engineering:** Industrialized experimental Jupyter research workflows into a reusable software structure, implementing binary object serialization using `joblib`.

### 🏆 Result
The optimized model recorded a decisive performance increase, escalating from an initial linear baseline of **66%** to a resilient cross-validated **77.14% R² score**, settling squarely into the target performance tier while ensuring strong generalization bounds.

---

## 🛠️ Model Performance Metrics
Evaluating the production-ready model yielded the following validation profile:

| Metric | Score / Value |
| :--- | :--- |
| **R² Score** | 77.14 % |
| **MAE** | *[Run your calculate_metrics on test and paste the € value here]* |
| **RMSE** | *[Run your calculate_metrics on test and paste the € value here]* |

---

## 📁 Repository Structure
```text
immo-eliza-ml/
│
├── data/
│   ├── raw/
│   └── cleaned/                   <-- Contains X_train_clean.csv & y_train_clean.csv
├── models/
│   └── xgboost_model.joblib       <-- Serialized production artifact
├── src/
│   ├── utils/
│   │   └── metrics_utils.py       <-- Metric calculation utilities
│   ├── train.py                   <-- Executable pipeline to train and dump the model
│   └── predict.py                 <-- Executable script for inference testing
├── requirements.txt               <-- Third-party library dependencies
└── README.md
```

---

## 🔧 Preparation for Immo-Eliza-ML - Sprint 3 of 4
```text
ImmoEliza-ml Project Organization and Execution Pipeline with content used for each step
│
├── 1. Data Splitting ─────────> Reference: 03-data-splitting.ipynb
│                               (Immediate isolation of Train and Test to secure the data)
│
├── 2. Data Preparation ───────> Reference: 01-preprocessing.ipynb
│                               (Imputer, Encoder, and Scaler: fit on Train, transform on Test)
│
├── 3. Baseline Model ─────────> Reference: 02-multiple-linear-regression.ipynb
│                               (Initial linear model and benchmark metrics)
│
├── 4. Model Comparison ───────> Reference: 03-regression-models-comparison.ipynb
│                               (Decision Trees, Random Forest, XGBoost)
│
└── 5. Optimization ───────────> Reference: 04-hyperparameter-tuning.ipynb & 02-overfitting.ipynb
                                 (GridSearchCV, Cross-Validation, Overfitting analysis)
```
