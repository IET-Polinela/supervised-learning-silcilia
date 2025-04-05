import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset_encoded (1).csv")

# Ambil fitur numerik dan target
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove("SalePrice")
X = df[numerical_features]
y = df["SalePrice"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
lr_pred = model_lr.predict(X_test)

# Evaluasi
mse_lr = mean_squared_error(y_test, lr_pred)
r2_lr = r2_score(y_test, lr_pred)

print(f"Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.4f}")
