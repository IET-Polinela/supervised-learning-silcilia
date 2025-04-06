import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Salin dataset tanpa outlier
df_scaled_std = df_no_outliers.copy()
df_scaled_minmax = df_no_outliers.copy()

# Scaling
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

df_scaled_std[numerical_features] = scaler_std.fit_transform(df_no_outliers[numerical_features])
df_scaled_minmax[numerical_features] = scaler_minmax.fit_transform(df_no_outliers[numerical_features])

# Plot histogram perbandingan (contoh: 6 fitur pertama)
sample_features = numerical_features[:6]

plt.figure(figsize=(20, 15))
for i, feature in enumerate(sample_features):
    plt.subplot(6, 3, i * 3 + 1)
    sns.histplot(df_no_outliers[feature], bins=30, kde=True, color='blue')
    plt.title(f"Original - {feature}")

    plt.subplot(6, 3, i * 3 + 2)
    sns.histplot(df_scaled_std[feature], bins=30, kde=True, color='green')
    plt.title(f"StandardScaler - {feature}")

    plt.subplot(6, 3, i * 3 + 3)
    sns.histplot(df_scaled_minmax[feature], bins=30, kde=True, color='orange')
    plt.title(f"MinMaxScaler - {feature}")

plt.tight_layout()
plt.savefig("scaling_histograms.png")
plt.show()
