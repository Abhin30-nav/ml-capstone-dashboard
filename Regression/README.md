# 23CSE301 Machine Learning — Capstone Project

## Project Overview
This repository contains the Machine Learning Capstone Project for course **23CSE301 Machine Learning** (AY 2026–2027). The project implements a complete, end-to-end Machine Learning pipeline across **Regression**, **Classification**, and **Clustering** tracks using the Tetouan City Power Consumption Dataset.

---

## Modular Notebooks Architecture

All data preprocessing, feature engineering, and 10 individual regression algorithms are modularized into separate, fully-executed `.ipynb` notebooks containing pre-rendered visualisations, score outputs, and markdown commentaries:

| Notebook File | Description |
| :--- | :--- |
| **`00_Data_Preprocessing_and_EDA.ipynb`** | Dataset audit, data cleaning, feature engineering, feature distribution plots, heatmap, scatter plots. |
| **`01_Linear_Regression.ipynb`** | Algorithm 1: Linear Regression baseline model & coefficient analysis. |
| **`02_Ridge_Regression.ipynb`** | Algorithm 2: Ridge Regression with $L_2$ regularisation penalty. |
| **`03_Lasso_Regression.ipynb`** | Algorithm 3: Lasso Regression with $L_1$ feature selection. |
| **`04_ElasticNet_Regression.ipynb`** | Algorithm 4: ElasticNet Regression (Combined $L_1 + L_2$). |
| **`05_Polynomial_Regression.ipynb`** | Algorithm 5: Polynomial Regression (Degree 2 feature expansion). |
| **`06_Decision_Tree_Regressor.ipynb`** | Algorithm 6: Decision Tree Regressor (Depth tuning & MSE splits). |
| **`07_Random_Forest_Regressor.ipynb`** | Algorithm 7: Random Forest Regressor ensemble baseline. |
| **`08_Gradient_Boosting_Regressor.ipynb`** | Algorithm 8: Gradient Boosting Regressor (Sequential boosting). |
| **`09_Support_Vector_Regressor.ipynb`** | Algorithm 9: Support Vector Regressor (Kernelized RBF SVR). |
| **`10_KNN_Regressor.ipynb`** | Algorithm 10: K-Nearest Neighbors Regressor (Distance-weighted). |
| **`11_Model_Comparison_and_Evaluation.ipynb`** | Consolidated 10-model ranking table, comparison chart, 5-fold CV, and `GridSearchCV` tuning. |
| **`Regression.ipynb`** | Comprehensive single-file notebook containing the complete end-to-end pipeline. |

---

## 10-Algorithm Comparison Summary

| Rank | Algorithm | $R^2$ Score | RMSE (kW) | MAE (kW) | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Regressor** | **0.8841** | **2,581.42** | **1,632.18** | 3.42s |
| **2** | **Gradient Boosting Regressor** | **0.8650** | **2,786.15** | **1,854.32** | 4.87s |
| **3** | **K-Nearest Neighbors (KNN)** | **0.8412** | **3,021.05** | **1,980.45** | 0.12s |
| **4** | **Decision Tree Regressor** | **0.8105** | **3,301.65** | **2,110.20** | 0.28s |
| **5** | **Support Vector Regressor (SVR)** | **0.7854** | **3,512.40** | **2,340.10** | 18.45s |
| **6** | **Polynomial Regression (deg=2)** | **0.6541** | **4,451.20** | **3,120.50** | 0.85s |
| **7** | **Ridge Regression** | **0.5218** | **5,231.10** | **3,890.40** | 0.05s |
| **8** | **Linear Regression (Baseline)** | **0.5218** | **5,231.15** | **3,890.45** | 0.04s |
| **9** | **Lasso Regression** | **0.5217** | **5,232.00** | **3,891.10** | 0.18s |
| **10** | **ElasticNet Regression** | **0.4852** | **5,425.80** | **4,112.30** | 0.12s |

---

## Directory Structure
```text
.
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies
├── powerconsumption 2.csv  # Raw dataset
├── Regression.ipynb        # Master single-file notebook
├── notebooks/              # Modularized separate notebooks directory
│   ├── 00_Data_Preprocessing_and_EDA.ipynb
│   ├── 01_Linear_Regression.ipynb
│   ├── 02_Ridge_Regression.ipynb
│   ├── 03_Lasso_Regression.ipynb
│   ├── 04_ElasticNet_Regression.ipynb
│   ├── 05_Polynomial_Regression.ipynb
│   ├── 06_Decision_Tree_Regressor.ipynb
│   ├── 07_Random_Forest_Regressor.ipynb
│   ├── 08_Gradient_Boosting_Regressor.ipynb
│   ├── 09_Support_Vector_Regressor.ipynb
│   ├── 10_KNN_Regressor.ipynb
│   └── 11_Model_Comparison_and_Evaluation.ipynb
├── data/
├── models/
└── app/
```
