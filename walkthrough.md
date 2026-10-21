# Project Walkthrough — Phase 2: Classification Part B & Clustering Track

## 1. Executive Summary

Phase 2 of the Capstone Project has been completely implemented, verified, and executed with zero errors in strict alignment with the course rubric (`23CSE301_ML_26_27_Capstone_Guidelines.pdf`).

Two dedicated, standalone Jupyter notebooks have been created, fully executed, and mirrored across the directory structure:
1. **`Classification_PartB.ipynb`** (and `Classification/Classification_PartB.ipynb`)
2. **`Clustering.ipynb`** (and `Clustering/Clustering.ipynb`)
3. **`Review2_Consolidated_Comparison.ipynb`** (and `Review2/Review2_Consolidated_Comparison.ipynb`)

---

## 2. Classification Track Part B: Ensemble Methods & Neural Networks

### Pipeline & Zero Data Leakage
- **Dataset:** Addis Ababa Road Traffic Accident (RTA) Severity (`12,316` samples, `31` features).
- **Target:** `Accident_severity` (`0: Fatal injury` [1.3%], `1: Serious Injury` [14.1%], `2: Slight Injury` [84.6%]).
- **Split:** Stratified 80:20 Train-Test split (`random_state=42`, `stratify=y`).
- **Scaling:** `StandardScaler` fitted strictly on `X_train` and applied to `X_test`.

### Part B Tuning Results (GridSearchCV with 5-Fold Cross-Validation on `f1_weighted`)
| Algorithm | Key Best Hyperparameters | Baseline F1 | Tuned F1 | Tuned Accuracy | Tuned ROC-AUC (OvR) | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `n_estimators=200, max_depth=None, min_samples_split=2` | 0.7786 | 0.7790 | 0.8474 | 0.6439 | 13.29s |
| **AdaBoost** | `learning_rate=1.0, n_estimators=50` | 0.7751 | 0.7751 | 0.8458 | 0.6171 | 6.22s |
| **Gradient Boosting** | `learning_rate=0.1, max_depth=5, n_estimators=150` | 0.7850 | **0.8019** | **0.8527** | **0.6814** | 41.49s |
| **Bagging (DT base)** | `n_estimators=50, max_samples=1.0, max_features=1.0` | 0.7865 | 0.7871 | 0.8498 | 0.6830 | 8.70s |
| **MLP Neural Net** | `hidden_layer_sizes=(128, 64), activation=relu, alpha=0.0001` | 0.7749 | 0.7766 | 0.8413 | 0.5918 | 3.35s |

### Consolidated Master Comparison Table Across All 10 Classification Algorithms (Review 2 Rubric A2)
| Rank | Algorithm Number | Model | Track Part | Accuracy | Precision (w) | Recall (w) | F1-Score (w) | ROC-AUC (OvR) | Fit Time (s) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **8** | **Gradient Boosting Classifier** | **Part B** | **0.8527** | **0.8253** | **0.8527** | **0.8019** | **0.6814** | 41.49s |
| **2** | **4** | **Decision Tree Classifier** | **Part A** | 0.8429 | 0.7935 | 0.8429 | 0.7959 | 0.6175 | 0.79s |
| **3** | **5** | **Support Vector Machine (SVC)** | **Part A** | 0.8287 | 0.7709 | 0.8287 | 0.7883 | 0.5734 | 93.99s |
| **4** | **9** | **Bagging Classifier (DT base)** | **Part B** | 0.8498 | 0.8219 | 0.8498 | 0.7871 | 0.6830 | 8.70s |
| **5** | **6** | **Random Forest Classifier** | **Part B** | 0.8474 | 0.8581 | 0.8474 | 0.7790 | 0.6439 | 13.29s |
| **6** | **10** | **MLP Classifier (Neural Net)** | **Part B** | 0.8413 | 0.7497 | 0.8413 | 0.7766 | 0.5918 | 3.35s |
| **7** | **2** | **K-Nearest Neighbors** | **Part A** | 0.8377 | 0.7431 | 0.8377 | 0.7753 | 0.5615 | 0.74s |
| **8** | **1** | **Logistic Regression** | **Part A** | 0.8458 | 0.7153 | 0.8458 | 0.7751 | 0.5801 | 3.86s |
| **9** | **7** | **AdaBoost Classifier** | **Part B** | 0.8458 | 0.7153 | 0.8458 | 0.7751 | 0.6171 | 6.22s |
| **10** | **3** | **Naive Bayes (Gaussian)** | **Part A** | 0.8141 | 0.7323 | 0.8141 | 0.7648 | 0.5648 | 0.22s |

---

## 3. Clustering Track: Unsupervised Social Media User Profiling

### Strict Unsupervised Rule Compliance
- **Dataset:** `social_media_usage.csv` (`1,000` users).
- **Features Used:** `Daily_Minutes_Spent`, `Posts_Per_Day`, `Likes_Per_Day`, `Follows_Per_Day`.
- Ground truth `App` and identifier `User_ID` were strictly **withheld from training** and only used post-hoc.
- Features standardized with `StandardScaler` for Euclidean distance compatibility.

### Mandatory Unsupervised Metrics Comparison Table (Review 2 Rubric B2)
| Algorithm | Key Hyperparameters | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Post-Hoc ARI vs App | Post-Hoc NMI vs App |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means Clustering** | `n_clusters=8, init='k-means++', n_init=10` | **0.2278** | **1.1305** | **228.52** | -0.0012 | 0.0087 |
| **Agglomerative Hierarchical** | `n_clusters=8, linkage='ward', metric='euclidean'` | 0.1596 | 1.4051 | 173.16 | 0.0019 | 0.0136 |

### Visualizations & Diagnostics Included (Review 2 Rubric B3)
1. **Elbow Curve (Inertia vs. $k$):** Identified sharp inflection around $k=8$.
2. **Silhouette Analysis Curve ($k=2..10$):** Reached global maximum at $k=8$ ($0.2278$).
3. **Hierarchical Dendrogram:** Ward linkage tree with horizontal cutoff threshold indicating 8 emergent cohorts.
4. **PCA 2D Projection:** 51.4% explained variance ratio with cluster color-coding for K-Means and Hierarchical.
5. **t-SNE 2D Manifold:** Non-linear cluster separation preserving local neighborhoods.
6. **Contingency Matrix Heatmap:** Cross-tabulation between discovered clusters and true social media apps.
7. **Discovered Personas:** Defined real-world engagement archetypes (*Super Users*, *Passive Lurkers*, *Active Networkers*, *Power Influencers*).

---

## 4. Verification & Validation Summary

- **`Classification_PartB.ipynb`:** Executed top-to-bottom via Jupyter nbconvert — **Exit code 0 (298 KB)**.
- **`Clustering.ipynb`:** Executed top-to-bottom via Jupyter nbconvert — **Exit code 0 (1.6 MB)**.
- **`Review2_Consolidated_Comparison.ipynb`:** Executed top-to-bottom via Jupyter nbconvert — **Exit code 0 (316 KB)**.
