import pandas as pd  # Pastikan ini ada di bagian awal

# Nilai evaluasi dari semua model
comparison_results = {
    "Linear Regression": {"MSE": 570837177.42, "R2": 0.8594},
    "Polynomial Degree 2": {"MSE": 20966191072.76, "R2": -4.2476},
    "Polynomial Degree 3": {"MSE": 691009110.83, "R2": 0.8270},
    "KNN (k=3)": {"MSE": 692648449.40, "R2": 0.8266},
    "KNN (k=5)": {"MSE": 510233027.09, "R2": 0.8723},
    "KNN (k=7)": {"MSE": 510227970.87, "R2": 0.8723}
}

comparison_df = pd.DataFrame(comparison_results).T
print("Perbandingan Model:")
print(comparison_df)
