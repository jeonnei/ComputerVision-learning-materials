from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터셋 로드
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 데이터셋 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 회귀 모델 정의 (k=5)
knn_regressor = KNeighborsRegressor(n_neighbors=5)

# 모델 학습
knn_regressor.fit(X_train, y_train)

# 예측
y_pred = knn_regressor.predict(X_test)

# MSE 및 R^2 점수 출력
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
