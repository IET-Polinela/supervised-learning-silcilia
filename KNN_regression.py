from sklearn.neighbors import KNeighborsRegressor

# Fungsi evaluasi KNN Regression
def evaluate_knn(k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_test, y_pred

# Evaluasi untuk K = 3, 5, 7
mse_knn3, r2_knn3, y_test3, y_pred3 = evaluate_knn(3)
mse_knn5, r2_knn5, y_test5, y_pred5 = evaluate_knn(5)
mse_knn7, r2_knn7, y_test7, y_pred7 = evaluate_knn(7)

# Visualisasi hasil prediksi KNN
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test3, y=y_pred3)
plt.title("KNN (K=3): Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test5, y=y_pred5)
plt.title("KNN (K=5): Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.subplot(1, 3, 3)
sns.scatterplot(x=y_test7, y=y_pred7)
plt.title("KNN (K=7): Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.tight_layout()

# Simpan hasil visualisasi
plt.savefig("knn_regression.png", dpi=300, bbox_inches='tight')
plt.show()

# Print hasil evaluasi
print(f"KNN (K=3) - MSE: {mse_knn3:.2f}, R2: {r2_knn3:.4f}")
print(f"KNN (K=5) - MSE: {mse_knn5:.2f}, R2: {r2_knn5:.4f}")
print(f"KNN (K=7) - MSE: {mse_knn7:.2f}, R2: {r2_knn7:.4f}")
