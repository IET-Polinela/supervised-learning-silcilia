import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
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

X = df_no_outliers[numerical_features]
y = df_no_outliers['SalePrice']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Polynomial Regression
def evaluate_polynomial_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_pred

# Polynomial Regression (Degree 3)
mse3, r2_3, y_pred3 = evaluate_polynomial_model(3)

# KNN Regression (K=5)
model_knn_5 = KNeighborsRegressor(n_neighbors=5)
model_knn_5.fit(X_train, y_train)
y_pred_knn_5 = model_knn_5.predict(X_test)

# Visualisasi hasil prediksi vs aktual
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_lr)
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

# Polynomial Regression (Degree 3)
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test, y=y_pred3)
plt.title("Polynomial Regression (Degree 3)")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

# KNN Regression (K=5)
plt.subplot(1, 3, 3)
sns.scatterplot(x=y_test, y=y_pred_knn_5)
plt.title("KNN Regression (K=5)")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.tight_layout()
plt.savefig("Analysis_Comparation_Models_and_Conclusion.png", dpi=300)
plt.show()
