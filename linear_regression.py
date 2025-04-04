import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load Dataset
df = pd.read_csv("train.csv")

# 2️⃣ Pisahkan fitur numerik dan kategori
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 3️⃣ Tangani Missing Values
imputer_num = SimpleImputer(strategy='mean')  # Mean untuk fitur numerik
imputer_cat = SimpleImputer(strategy='most_frequent')  # Mode untuk fitur kategori

df[numerical_features] = imputer_num.fit_transform(df[numerical_features])
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# 4️⃣ Encoding Fitur Kategori
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(df[categorical_features])
df_encoded = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

# Gabungkan dataset yang sudah di-encode
df_final = pd.concat([df[numerical_features], df_encoded], axis=1)

# 5️⃣ Menghapus Outlier dengan IQR
def remove_outliers_iqr(data, features):
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[features] >= lower_bound) & (data[features] <= upper_bound)].dropna()

df_no_outliers = remove_outliers_iqr(df_final, numerical_features)

# 6️⃣ Simpan Dataset yang Sudah Diproses
df_no_outliers.to_csv("dataset_without_outliers.csv", index=False)
df_final.to_csv("dataset_encoded.csv", index=False)

# 7️⃣ Persiapan Data untuk Model
X_outlier = df_final.drop(columns=["SalePrice"])  # Dataset dengan outlier
y_outlier = df_final["SalePrice"]

X_cleaned = df_no_outliers.drop(columns=["SalePrice"])  # Dataset tanpa outlier
y_cleaned = df_no_outliers["SalePrice"]

# 8️⃣ Split Data
X_train_outlier, X_test_outlier, y_train_outlier, y_test_outlier = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# 9️⃣ Train Model
model_outlier = LinearRegression()
model_cleaned = LinearRegression()

model_outlier.fit(X_train_outlier, y_train_outlier)
model_cleaned.fit(X_train_cleaned, y_train_cleaned)

# 🔟 Prediksi
y_pred_outlier = model_outlier.predict(X_test_outlier)
y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

# 1️⃣1️⃣ Evaluasi Model
mse_outlier = mean_squared_error(y_test_outlier, y_pred_outlier)
r2_outlier = r2_score(y_test_outlier, y_pred_outlier)

mse_cleaned = mean_squared_error(y_test_cleaned, y_pred_cleaned)
r2_cleaned = r2_score(y_test_cleaned, y_pred_cleaned)

print("🔹 Dataset dengan Outlier 🔹")
print(f"📌 MSE: {mse_outlier:.2f}")
print(f"📌 R² Score: {r2_outlier:.4f}")

print("\n🔹 Dataset Tanpa Outlier 🔹")
print(f"📌 MSE: {mse_cleaned:.2f}")
print(f"📌 R² Score: {r2_cleaned:.4f}")

# 1️⃣2️⃣ Visualisasi Hasil
plt.figure(figsize=(15, 5))

# Scatter Plot: Prediksi vs Aktual
plt.subplot(1, 3, 1)
plt.scatter(y_test_outlier, y_pred_outlier, alpha=0.5, label="Dengan Outlier", color="red")
plt.scatter(y_test_cleaned, y_pred_cleaned, alpha=0.5, label="Tanpa Outlier", color="blue")
plt.plot([min(y_test_outlier), max(y_test_outlier)], [min(y_test_outlier), max(y_test_outlier)], 'k--')
plt.xlabel("Nilai Aktual")
plt.ylabel("Prediksi")
plt.legend()
plt.title("Scatter Plot: Prediksi vs Aktual")

# Residual Plot
plt.subplot(1, 3, 2)
sns.histplot(y_test_outlier - y_pred_outlier, bins=50, color="red", alpha=0.6, label="Dengan Outlier", kde=True)
sns.histplot(y_test_cleaned - y_pred_cleaned, bins=50, color="blue", alpha=0.6, label="Tanpa Outlier", kde=True)
plt.xlabel("Residual")
plt.ylabel("Frekuensi")
plt.legend()
plt.title("Distribusi Residual")

# Residual vs Prediksi
plt.subplot(1, 3, 3)
plt.scatter(y_pred_outlier, y_test_outlier - y_pred_outlier, alpha=0.5, label="Dengan Outlier", color="red")
plt.scatter(y_pred_cleaned, y_test_cleaned - y_pred_cleaned, alpha=0.5, label="Tanpa Outlier", color="blue")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Prediksi")
plt.ylabel("Residual")
plt.legend()
plt.title("Residual vs Prediksi")

# Simpan hasil visualisasi sebagai PNG
plt.savefig("linear_regression1.png", dpi=300, bbox_inches='tight')
plt.show()
