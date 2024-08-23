from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터셋 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 분류 모델 정의 (k=3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# 모델 학습
knn_classifier.fit(X_train, y_train)

# 예측
y_pred = knn_classifier.predict(X_test)

# 정확도 및 분류 보고서 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
