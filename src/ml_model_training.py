import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv("output/ml_dataset.csv")



# Drop date/id and prepare target
# Drop date/id and prepare target
# Drop date/id and prepare target
X = df.drop(columns=["fwd_1m_ret", "date", "id"], errors="ignore")
y = df["fwd_1m_ret"]
print("Target preview:")
print(y.describe())
print("Any NaNs?", y.isna().sum())
print("Any zero variance?", y.nunique())

# Remove extreme outliers in target
# Remove extreme outliers in target
y_std = y.std()
y_mean = y.mean()
is_outlier = (y < y_mean - 5 * y_std) | (y > y_mean + 5 * y_std)

X = X[~is_outlier]
y = y[~is_outlier]

# Sanity check to avoid zero-sample error
if len(X) == 0:
    raise ValueError("No data left after outlier filtering — adjust thresholds or inspect fwd_1m_ret distribution.")


# Ensure string column names
X.columns = X.columns.astype(str)

# Normalize features
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
joblib.dump(scaler, "output/scaler.joblib")

# Re-wrap scaled features into DataFrame with string column names
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# X_train now has correct feature names as strings!


# Time series split
tscv = TimeSeriesSplit(n_splits=5)

# Grid Search
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0]

}

model = XGBRegressor(objective="reg:squarederror", random_state=42)

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Params: {grid_search.best_params_}")
print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test R^2: {r2:.4f}")

# Save model
os.makedirs("output", exist_ok=True)
joblib.dump(model, "output/ml_model.joblib")
print("Trained model saved to output/ml_model.joblib")