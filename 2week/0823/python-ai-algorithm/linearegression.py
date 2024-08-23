import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 생성
np.random.seed(0)
X = np.random.rand(100, 1)  # 100개의 샘플, 1개의 feature
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1  # y = 3x + 2 + 잡음

# 모델 정의
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# 모델 컴파일
model.compile(optimizer='sgd', loss='mse')

# 모델 학습
model.fit(X, y, epochs=100, verbose=1)

# 예측
pred = model.predict(X)
