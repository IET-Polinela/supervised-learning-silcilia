import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "train.csv"  # Sesuaikan path jika berbeda
df = pd.read_csv(file_path)

# Menampilkan informasi awal dataset
print(df.info())
print(df.describe())

# Identifikasi outlier menggunakan metode IQR
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]

# Identifikasi outlier untuk semua fitur numerik
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_counts = {feature: detect_outliers_iqr(df, feature).shape[0] for feature in numerical_features}

# Menampilkan jumlah outlier per fitur
print("Jumlah Outlier per Fitur (Metode IQR):")
for feature, count in outlier_counts.items():
    print(f"{feature}: {count}")

# Menghapus outlier dari dataset
def remove_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

df_no_outliers = df.copy()
for feature in numerical_features:
    df_no_outliers = remove_outliers(df_no_outliers, feature)

# Visualisasi Boxplot Sebelum dan Sesudah Penghapusan Outlier
plt.figure(figsize=(20, 60))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features)//4 + 1, 4, i)
    sns.boxplot(x=df[col], color='green')
    plt.title(f"Before - {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 60))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features)//4 + 1, 4, i)
    sns.boxplot(x=df_no_outliers[col])
    plt.title(f"After - {col}")
plt.tight_layout()
plt.savefig("outlier_handling.png")
plt.show()

# Visualisasi Histogram Sebelum & Sesudah Penghapusan Outlier (contoh pada fitur pertama)
plt.figure(figsize=(12, 5))
feature_sample = numerical_features[0]
sns.histplot(df[feature_sample], bins=30, color='red', kde=True, label="Before")
sns.histplot(df_no_outliers[feature_sample], bins=30, color='blue', kde=True, label="After")
plt.title(f"Distribusi {feature_sample} Sebelum & Sesudah Penghapusan Outlier")
plt.legend()
plt.show()

# Output jumlah data sebelum dan sesudah penghapusan outlier
print("Sebelum penghapusan outlier:", df.shape)
print("Sesudah penghapusan outlier:", df_no_outliers.shape)
