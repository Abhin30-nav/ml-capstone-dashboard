# 23CSE301 Machine Learning — Capstone Project

## Project Overview
This repository contains the Machine Learning Capstone Project for course **23CSE301 Machine Learning** (AY 2026–2027). The project implements complete, end-to-end Machine Learning pipelines across **Regression**, **Classification**, and **Clustering** tracks using the Tetouan City Power Consumption Dataset, the Addis Ababa Road Traffic Accident (RTA) Dataset, and the Social Media Usage Dataset.

---

## 1. Classification Track: Consolidated 10-Algorithm Comparison (Review 2 Deliverable)

### Dataset: Road Traffic Accident (RTA) Dataset (Addis Ababa Police Dept, 2017–2020)
### Target Variable: `Accident_severity` (`Slight Injury`, `Serious Injury`, `Fatal injury`)
### Methodology: Stratified 80:20 Train-Test split (`random_state=42`), `StandardScaler` (fitted on train only — Zero Data Leakage), 5-Fold Cross-Validation GridSearchCV on `f1_weighted`.

| Rank | Algorithm Number | Model | Track Part | Best Hyperparameters | Accuracy | Precision (w) | Recall (w) | F1-Score (w) | ROC-AUC (OvR) | Fit Time (s) |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **8** | **Gradient Boosting Classifier** | **Part B** | `learning_rate=0.1, max_depth=5, n_estimators=150` | **0.8527** | **0.8253** | **0.8527** | **0.8019** | **0.6814** | 41.49s |
| **2** | **4** | **Decision Tree Classifier** | **Part A** | `max_depth=8, criterion=gini, min_samples_split=5` | 0.8429 | 0.7935 | 0.8429 | 0.7959 | 0.6175 | 0.79s |
| **3** | **5** | **Support Vector Machine (SVC)** | **Part A** | `C=10, kernel=rbf` | 0.8287 | 0.7709 | 0.8287 | 0.7883 | 0.5734 | 93.99s |
| **4** | **9** | **Bagging Classifier (DT base)** | **Part B** | `n_estimators=50, max_samples=1.0, max_features=1.0` | 0.8498 | 0.8219 | 0.8498 | 0.7871 | 0.6830 | 8.70s |
| **5** | **6** | **Random Forest Classifier** | **Part B** | `n_estimators=200, max_depth=None, min_samples_split=2` | 0.8474 | 0.8581 | 0.8474 | 0.7790 | 0.6439 | 13.29s |
| **6** | **10** | **MLP Classifier (Neural Net)** | **Part B** | `hidden_layer_sizes=(128, 64), activation=relu, alpha=0.0001` | 0.8413 | 0.7497 | 0.8413 | 0.7766 | 0.5918 | 3.35s |
| **7** | **2** | **K-Nearest Neighbors** | **Part A** | `n_neighbors=7, weights=distance` | 0.8377 | 0.7431 | 0.8377 | 0.7753 | 0.5615 | 0.74s |
| **8** | **1** | **Logistic Regression** | **Part A** | `C=0.01, solver=lbfgs` | 0.8458 | 0.7153 | 0.8458 | 0.7751 | 0.5801 | 3.86s |
| **9** | **7** | **AdaBoost Classifier** | **Part B** | `learning_rate=1.0, n_estimators=50` | 0.8458 | 0.7153 | 0.8458 | 0.7751 | 0.6171 | 6.22s |
| **10** | **3** | **Naive Bayes (Gaussian)** | **Part A** | `var_smoothing=1e-06` | 0.8141 | 0.7323 | 0.8141 | 0.7648 | 0.5648 | 0.22s |

> **Best Model Selection Justification (Rubric A3):**  
> Under extreme multi-class imbalance (~84.6% slight injury, 14.1% serious injury, 1.3% fatal injury), **Gradient Boosting Classifier** achieved the highest overall Accuracy (0.8527), weighted F1-Score (0.8019), and multi-class ROC-AUC (0.6814). Its stage-wise optimization of multinomial deviance effectively captures complex non-linear interactions without succumbing to majority-class dominance.

---

## 2. Clustering Track: Unsupervised Social Media User Profiling (Review 2 Deliverable)

### Dataset: Social Media Usage Dataset (`social_media_usage.csv`, 1,000 active user profiles)
### Features: `Daily_Minutes_Spent`, `Posts_Per_Day`, `Likes_Per_Day`, `Follows_Per_Day`
### Ground-Truth Rule Compliance:
Labels (`App`) and `User_ID` were strictly **excluded during model fitting**. Clustering operated purely on standardized numerical metrics. Ground truth was utilized solely post-hoc for interpretation and validation.

| Algorithm | Key Hyperparameters | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Post-Hoc ARI vs App | Post-Hoc NMI vs App |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means Clustering** | `n_clusters=8, init='k-means++', n_init=10` | **0.2278** | **1.1305** | **228.52** | -0.0012 | 0.0087 |
| **Agglomerative Hierarchical** | `n_clusters=8, linkage='ward', metric='euclidean'` | 0.1596 | 1.4051 | 173.16 | 0.0019 | 0.0136 |

- **Optimal $k$ Selection:** Identified at $k=8$ via Elbow Method (Inertia plateau) and Silhouette Score global peak.
- **Linkage Strategy Comparison:** Ward linkage (Silhouette: 0.1596) significantly outperformed Average (0.1399), Complete (0.1361), and Single (-0.3695).
- **Dimensionality Reduction:** Evaluated via PCA (2 components, 51.4% explained variance) and t-SNE 2D manifold.
- **Behavioral Interpretation:** Discovered genuine user engagement personas (*Power Influencers*, *Passive Lurkers*, *Active Networkers*, *Super Users*) that naturally transcend specific social media apps.

---

## 3. Regression Track (Review 1 Scope)

### Dataset: Tetouan City Power Consumption Dataset (`powerconsumption 2.csv`, 52,416 rows)
### Target Variable: `PowerConsumption_Zone1`
### Tuning: GridSearchCV (5-fold CV, scoring=`r2`)

| Rank | Algorithm | Best Params | Tuned R² | RMSE (kW) | MAE (kW) | 5-fold CV R² | Fit Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **KNN Regressor** | `n_neighbors=5, weights=distance` | **0.9840** | 897.60 | 545.82 | 0.7007 ± 0.23 | 1.60s |
| **2** | **Random Forest Regressor** | `max_depth=20, n_estimators=150` | 0.9781 | 1052.09 | 677.83 | 0.7441 ± 0.22 | 111.75s |
| **3** | **Gradient Boosting Regressor** | `lr=0.1, max_depth=6, n_estimators=200` | 0.9654 | 1321.20 | 936.41 | — | 122.39s |
| **4** | **Decision Tree Regressor** | `max_depth=12, min_samples_split=5` | 0.9513 | 1567.65 | 1025.92 | — | 5.41s |
| **5** | **Polynomial Regression** | N/A (degree=2 fixed) | 0.8961 | 2290.96 | 1733.25 | — | 0.05s |

*5-fold CV R² is computed on the full dataset for the two best-performing models (KNNRegressor and RandomForestRegressor).*

---

## 4. Repository & Folder Architecture

```text
.
├── Classification_PartB.ipynb                     ← Phase 2: Classification Part B (Algos 6-10 + 10-Algo table)
├── Clustering.ipynb                               ← Phase 2: Clustering Track (K-Means, Hierarchical, PCA, t-SNE)
├── Review2_Consolidated_Comparison.ipynb          ← Master Review 2 Deliverable (All tracks consolidated)
│
├── Classification/
│   ├── RTA Dataset.csv                            ← Raw Classification Dataset
│   ├── 00_Data_Preprocessing_and_EDA.ipynb        ← EDA & Preprocessing Pipeline
│   ├── 01_Logistic_Regression.ipynb ... 05_Support_Vector_Classifier.ipynb (Part A notebooks)
│   ├── 06_Classification_PartA_Comparison.ipynb  ← Part A Preliminary Comparison
│   ├── Classification_PartB.ipynb                 ← Mirrored Part B Deliverable
│   └── Classification.ipynb                       ← Master Classification Pipeline
│
├── Clustering/
│   ├── social_media_usage.csv                     ← Raw Clustering Dataset
│   └── Clustering.ipynb                           ← Mirrored Clustering Deliverable
│
├── Regression/
│   ├── powerconsumption 2.csv                     ← Raw Regression Dataset
│   ├── 00_Data_Preprocessing_and_EDA.ipynb
│   ├── 01_Linear_Regression.ipynb ... 10_KNN_Regressor.ipynb
│   ├── 11_Model_Comparison_and_Evaluation.ipynb  ← Regression Comparative Evaluation
│   └── Regression.ipynb                           ← Master Regression Pipeline
│
├── Review2/
│   └── Review2_Consolidated_Comparison.ipynb      ← Mirrored Review 2 Deliverable
│
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf      ← Official Rubrics and Guidelines
├── Project_Teams_G.xlsx                           ← Team and Dataset Allocations
├── requirements.txt                               ← Python environment dependencies
├── walkthrough.md                                 ← Complete Phase 2 implementation summary
└── README.md
```

---

## 5. Instructions to Run

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Deliverable Notebooks:**
   - **Classification Part B:** Open and run `Classification_PartB.ipynb`
   - **Clustering Track:** Open and run `Clustering.ipynb`
   - **Review 2 Master Consolidated Notebook:** Open and run `Review2_Consolidated_Comparison.ipynb`
   All cells run top-to-bottom with zero execution errors.
