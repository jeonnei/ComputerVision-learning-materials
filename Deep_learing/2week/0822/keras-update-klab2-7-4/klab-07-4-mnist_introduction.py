from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical  # Updated import
import numpy as np
np.random.seed(777)  # for reproducibility

from keras.datasets import mnist

nb_classes = 10

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, nb_classes)  # Updated to use `to_categorical`
y_test = to_categorical(y_test, nb_classes)

model = Sequential()
# MNIST data image of shape 28 * 28 = 784
model.add(Dense(nb_classes, input_dim=784))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('\nAccuracy:', score[1])

'''
Expected Accuracy: Around 0.91 - 0.92
'''
