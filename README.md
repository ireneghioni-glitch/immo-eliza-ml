# Immo-Eliza Machine Learning Model 🚀

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

## ⭐️ Project Summary

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

## 🧮 Model Performance Metrics
Evaluating the production-ready model yielded the following validation profile:

| Metric | Score / Value |
| :--- | :--- |
| **R² Score** | 77.14 % |
| **MAE** | *[Run your calculate_metrics on test and paste the € value here]* |
| **RMSE** | *[Run your calculate_metrics on test and paste the € value here]* |

---

## 📊 Model Evaluation & Comparison
A rigorous benchmarking process was conducted across multiple algorithms. The table and chart below illustrate the performance leap achieved by transitioning from linear baselines to ensemble methods, combined with strategic outlier mitigation:

![Model Comparison](images/charts/model_comparison_results.png)

### Key Takeaways:
* **The Ensemble Advantage:** Tree-based models (`XGBoost` and `Random Forest`) significantly outperformed linear models, capturing the non-linear dynamics of the Belgian real estate market.
* **Impact of Outlier Removal:** Cleaning the top and bottom 1% price anomalies stabilized the gradient boosting process, pushing the final XGBoost Test $R^2$ to **77.14%**, well within the target performance tier.

---

## 🔍 Feature Importance (Model Explainability)
To understand what drives real estate prices in the Belgian market, the top 10 most impactful features extracted by the optimized XGBoost model were analyzed:

![Feature Importance](images/plots/top_10_most_relevant_feat.png)

### Key Insights from the Model:
* **Living Area & Property Structure:** The overall size of the property remains the strongest driver for valuation.
* **Geographical Impacts:** Regional and provincial encoding weights confirm that location drastically shifts the baseline price criteria across Belgium.

---

## 🚀 How to Run

### Installation  
Deploy dependencies into your virtual environment:

`pip install -r requirements.txt`

### Training Pipeline
Re-run the training process and serialize the model architecture:

`python src/train.py`

### Inference Engine
Generate predictions against target dummy data shapes:

`python src/predict.py`

---

## 📁 Repository Structure
```text
immo-eliza-ml/
│
├── data/
│   ├── raw/                              <-- Contains properties_final_irene.csv and other original datasets used
│   │   ├── train_test                    <-- Contains datasets after splitting
│   │   └── StatBel                       <-- Contains datasets from StatBel used in feature engineering phase
│   │    
│   └── cleaned/                          <-- Contains splitted and preprocessed datasets for training:
│                                             X_train_clean.csv & y_train_clean.csv
├── images/                        
│   ├── charts/
│   │   └── model_comparison_results.png  <-- Comparison table with obtained metrics foer each model
│   └── plots/
│       ├── decision_tree_vis.png         <-- Decision tree visualization
│       └── top_10_most_relevant_feat.png <-- Top 10 most relevant features bar plot
├── models/
│   └── xgboost_model.joblib              <-- Serialized production artifact
├── src/
│   ├── utils/
│   │   └── metrics_utils.py              <-- Metric calculation utilities
│   ├── train.py                          <-- Executable pipeline to train and dump the model
│   └── predict.py                        <-- Executable script for inference testing
├── requirements.txt                      <-- Third-party library dependencies
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
