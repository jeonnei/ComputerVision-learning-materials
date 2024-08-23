import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 생성
np.random.seed(0)
X = np.random.randn(100, 1)
y = (X > 0).astype(int)  # 0 또는 1로 변환

# 모델 정의
model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X, y, epochs=100, verbose=1)

# 예측
pred = model.predict(X)
