 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Hapus outlier dari semua fitur numerik
def remove_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove("SalePrice")
df_no_outliers = df.copy()
for feature in numerical_features:
    df_no_outliers = remove_outliers(df_no_outliers, feature)

# Pisahkan fitur (X) dan target (y)
X = df_no_outliers[numerical_features]
y = df_no_outliers['SalePrice']

# Scaling fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fungsi evaluasi model Polynomial Regression
def evaluate_polynomial_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Evaluasi Overfitting
    r2_train = r2_score(y_train, model.predict(X_poly_train))

    return mse, r2, r2_train, y_test, y_pred

# Evaluasi Linear Regression sebagai baseline
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Evaluasi Polynomial Regression dengan degree 2
mse2, r2_2, r2_train_2, y_test2, y_pred2 = evaluate_polynomial_model(2)

# Evaluasi Polynomial Regression dengan degree 3
mse3, r2_3, r2_train_3, y_test3, y_pred3 = evaluate_polynomial_model(3)

# Print hasil evaluasi
print("=== Model Evaluation ===")
print(f"Linear Regression - MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}")
print(f"Polynomial Degree 2 - MSE: {mse2:.2f}, R2 Test: {r2_2:.4f}, R2 Train: {r2_train_2:.4f}")
print(f"Polynomial Degree 3 - MSE: {mse3:.2f}, R2 Test: {r2_3:.4f}, R2 Train: {r2_train_3:.4f}")

# Visualisasi Predicted vs Actual
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_lin)
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test2, y=y_pred2)
plt.title("Polynomial Degree 2: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.subplot(1, 3, 3)
sns.scatterplot(x=y_test3, y=y_pred3)
plt.title("Polynomial Degree 3: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.tight_layout()

# Simpan gambar visualisasi Predicted vs Actual
plt.savefig("polynomial_regression.png", dpi=300, bbox_inches='tight')
plt.show()

# Visualisasi distribusi residuals
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(y_test - y_pred_lin, bins=30, kde=True)
plt.title("Residuals Distribution: Linear Regression")

plt.subplot(1, 3, 2)
sns.histplot(y_test2 - y_pred2, bins=30, kde=True)
plt.title("Residuals Distribution: Degree 2")

plt.subplot(1, 3, 3)
sns.histplot(y_test3 - y_pred3, bins=30, kde=True)
plt.title("Residuals Distribution: Degree 3")

plt.tight_layout()
plt.show()
