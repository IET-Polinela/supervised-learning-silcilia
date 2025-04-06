# 📌 Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️ Load Dataset
df = pd.read_csv("train.csv")

# Solusi: Ubah semua nama kolom menjadi string
df.columns = df.columns.astype(str)

# 2 Pisahkan fitur numerik dan kategori
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# 3️ Imputasi (Mengisi NaN)
imputer_num = SimpleImputer(strategy="mean")
imputer_cat = SimpleImputer(strategy="most_frequent")

df[numerical_features] = imputer_num.fit_transform(df[numerical_features])
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# 4️ Encoding fitur kategori
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
df_encoded[numerical_features] = df[numerical_features].reset_index(drop=True)

# 🔥 Solusi: Ubah kembali semua nama kolom setelah encoding
df_encoded.columns = df_encoded.columns.astype(str)

# 5️⃣ Fungsi untuk menghapus outlier dengan IQR
def remove_outliers_iqr(data, features, threshold=1.5):
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (threshold * IQR)
    upper_bound = Q3 + (threshold * IQR)
    return data[~((data[features] < lower_bound) | (data[features] > upper_bound)).any(axis=1)]

# 6️⃣ Hapus outlier
df_no_outliers = remove_outliers_iqr(df_encoded, numerical_features)

# 7️⃣ Simpan dataset yang telah diproses
df_encoded.to_csv("dataset_encoded.csv", index=False)  # Dataset setelah encoding
df_no_outliers.to_csv("dataset_without_outliers.csv", index=False)  # Dataset setelah menghapus outlier

# 8️⃣ Scaling data
scaler = StandardScaler()

X_outlier = df_encoded.drop(columns=["SalePrice"])
y_outlier = df_encoded["SalePrice"]

X_cleaned = df_no_outliers.drop(columns=["SalePrice"])
y_cleaned = df_no_outliers["SalePrice"]

X_outlier_scaled = pd.DataFrame(scaler.fit_transform(X_outlier), columns=X_outlier.columns)
X_cleaned_scaled = pd.DataFrame(scaler.fit_transform(X_cleaned), columns=X_cleaned.columns)

# 🔟 Train & Evaluate Model
def train_and_evaluate(X_train, X_test, y_train, y_test, title, filename):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"🔹 {title} 🔹")
    print(f"📌 MSE: {mse:.2f}")
    print(f"📌 R² Score: {r2:.4f}")
    print("-" * 40)

    # 📊 Scatter Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Scatter Plot ({title})")

    # 📊 Residual Plot
    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({title})")

    # 📊 Distribusi Residual
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title(f"Distribusi Residual ({title})")

    # Simpan gambar
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# 🔥 Split data menjadi train & test
X_train_outlier, X_test_outlier, y_train_outlier, y_test_outlier = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_cleaned_scaled, y_cleaned, test_size=0.2, random_state=42)

# 🔥 Evaluasi Model
train_and_evaluate(X_train_outlier, X_test_outlier, y_train_outlier, y_test_outlier, "Dataset dengan Outlier", "Linear_Regression_Outlier.png")
train_and_evaluate(X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned, "Dataset Tanpa Outlier", "Linear_Regression_Cleaned.png")
train_and_evaluate(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, "Dataset Tanpa Outlier & Scaling", "Linear_Regression_Scaled.png")
 
