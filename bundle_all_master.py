import joblib
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
     Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

print('⏳ Training and bundling all master capstone models (Sequential Mode)...')

# --- 1. CLASSIFICATION TRACK (RTA Dataset) ---
csv_path = (
    'RTA Dataset.csv'
    if os.path.exists('RTA Dataset.csv')
    else 'Classification/RTA Dataset.csv'
)
df_raw = pd.read_csv(csv_path)

df_clean = df_raw.copy()
for col in df_clean.columns:
  if not pd.api.types.is_numeric_dtype(df_clean[col]):
    df_clean[col] = df_clean[col].fillna('Unknown')

df_clean['Hour'] = (
    pd.to_datetime(df_clean['Time'], format='%H:%M:%S', errors='coerce')
    .dt.hour.fillna(12)
    .astype(int)
)
df_clean['Is_Weekend'] = df_clean['Day_of_week'].isin(
    ['Saturday', 'Sunday']
).astype(int)
df_clean['Casualties_Per_Vehicle'] = (
    df_clean['Number_of_casualties']
    / df_clean['Number_of_vehicles_involved'].replace(0, 1)
).round(2)

drop_cols = ['Time', 'Accident_severity']
numerical_features = [
    'Number_of_vehicles_involved',
    'Number_of_casualties',
    'Hour',
    'Casualties_Per_Vehicle',
]
categorical_features = [
    c for c in df_clean.columns if c not in drop_cols + numerical_features
]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    (
        'encoder',
        OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ),
    ),
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features),
])

target_encoder = LabelEncoder()
y_class = target_encoder.fit_transform(df_clean['Accident_severity'])
X_class = df_class = df_clean[numerical_features + categorical_features]
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_class, y_class, test_size=0.20, random_state=42, stratify=y_class
)

classification_models = {
    'Gradient Boosting (Class)': Pipeline([
        ('prep', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42)),
    ]).fit(X_tr_c, y_tr_c),
    'Random Forest (Class)': Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=None)),
    ]).fit(X_tr_c, y_tr_c),
    'Bagging Classifier': Pipeline([
        ('prep', preprocessor),
        ('clf', BaggingClassifier(random_state=42, n_jobs=None)),
    ]).fit(X_tr_c, y_tr_c),
    'AdaBoost': Pipeline([
        ('prep', preprocessor),
        ('clf', AdaBoostClassifier(random_state=42)),
    ]).fit(X_tr_c, y_tr_c),
    'MLP Neural Net': Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=300, random_state=42)),
    ]).fit(X_tr_c, y_tr_c),
    'Decision Tree (Class)': Pipeline([
        ('prep', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=42)),
    ]).fit(X_tr_c, y_tr_c),
    'SVM (Class)': Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42)),
    ]).fit(X_tr_c[:3000], y_tr_c[:3000]),
    'Logistic Regression': Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, random_state=42)),
    ]).fit(X_tr_c, y_tr_c),
    'KNN (Class)': Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_jobs=None)),
    ]).fit(X_tr_c, y_tr_c),
    'Naive Bayes': Pipeline([
        ('prep', preprocessor),
        ('clf', GaussianNB()),
    ]).fit(X_tr_c, y_tr_c),
}

# --- 2. REGRESSION TRACK ---
reg_file = (
    'powerconsumption 2.csv'
    if os.path.exists('powerconsumption 2.csv')
    else 'data/powerconsumption.csv'
)
reg_df = pd.read_csv(reg_file)

X_reg = reg_df[[
    'Temperature',
    'Humidity',
    'WindSpeed',
    'GeneralDiffuseFlows',
    'DiffuseFlows',
]].copy()
y_reg = reg_df['PowerConsumption_Zone1']

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

regression_models = {
    'Linear Regression': (
        LinearRegression().fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
    'Ridge Regression': (
        Ridge(random_state=42).fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
    'Lasso Regression': (
        Lasso(random_state=42).fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
    'ElasticNet': (
        ElasticNet(random_state=42).fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
    'Decision Tree Regressor': (
        DecisionTreeRegressor(max_depth=8, random_state=42).fit(
            X_reg_scaled, y_reg
        ),
        scaler_reg,
    ),
    'Random Forest Regressor': (
        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=None).fit(
            X_reg_scaled, y_reg
        ),
        scaler_reg,
    ),
    'Gradient Boosting Regressor': (
        GradientBoostingRegressor(random_state=42).fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
    
    'K-Neighbors Regressor': (
        KNeighborsRegressor(n_jobs=None).fit(X_reg_scaled, y_reg),
        scaler_reg,
    ),
}

# --- 3. CLUSTERING TRACK ---
clus_file = (
    'social_media_usage.csv'
    if os.path.exists('social_media_usage.csv')
    else 'Clustering/social_media_usage.csv'
)
clus_df = pd.read_csv(clus_file)
X_clus = clus_df[[
    'Daily_Minutes_Spent',
    'Posts_Per_Day',
    'Likes_Per_Day',
    'Follows_Per_Day',
]]
scaler_clus = StandardScaler()
X_clus_scaled = scaler_clus.fit_transform(X_clus)

kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10).fit(
    X_clus_scaled
)
clustering_bundle = {'K-Means': kmeans_model, 'scaler': scaler_clus}

# Master Bundle Dictionary
master_capstone_dict = {
    'Classification': classification_models,
    'Regression': regression_models,
    'Clustering': clustering_bundle,
}

# --- 4. SAVE INTO SEPARATE TRACK FILES ---
# Save into separate track files with maximum compression
joblib.dump(master_capstone_dict["Regression"], "regression_models.pkl", compress=9)
joblib.dump(master_capstone_dict["Classification"], "classification_models.pkl", compress=9)
joblib.dump(master_capstone_dict["Clustering"], "clustering_models.pkl", compress=9)

print("✅ Success! Compressed models saved under 100MB.")