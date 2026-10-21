import joblib
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 1. Load Data
csv_file = (
    'powerconsumption 2.csv'
    if os.path.exists('powerconsumption 2.csv')
    else 'data/powerconsumption.csv'
)
df = pd.read_csv(csv_file)

# Feature Engineering (Temporal features matching notebook guidelines)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Define Features and Target
X = df.drop(
    columns=[
        'Datetime',
        'PowerConsumption_Zone1',
        'PowerConsumption_Zone2',
        'PowerConsumption_Zone3',
    ]
)
y = df['PowerConsumption_Zone1']

# 80:20 Train-Test Split with strict zero data leakage scaling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define All 10 Regression Algorithms
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=50, max_depth=12, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    'Support Vector Regressor (SVR)': SVR(C=100, epsilon=0.1),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}

trained_models = {}
for name, model in models.items():
  print(f'Training {name}...')
  if name in ['Support Vector Regressor (SVR)', 'K-Nearest Neighbors']:
    model.fit(X_train_scaled[:5000], y_train[:5000])  # Subsample for speed if needed
    trained_models[name] = (model, scaler, True)
  else:
    model.fit(X_train_scaled, y_train)
    trained_models[name] = (model, scaler, True)

joblib.dump(trained_models, 'all_regression_models.pkl')
print('All regression models successfully saved to all_regression_models.pkl!')