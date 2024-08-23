from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical  # Updated import
import numpy as np

# Predicting animal type based on various features
# Update the file path according to your environment
xy = np.loadtxt('C:/Users/kchan/python-Deeplearning/DeepLearningZeroToAll/keras/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]].astype(int) - 1  # Convert y_data to int
print(x_data.shape, y_data.shape)

nb_classes = 7
y_one_hot = to_categorical(y_data, nb_classes)  # Updated to use `to_categorical`

model = Sequential()
model.add(Dense(nb_classes, input_shape=(16,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_one_hot, epochs=1000)

# Updated prediction logic (use np.argmax with predict)
pred = np.argmax(model.predict(x_data), axis=-1)
for p, y in zip(pred, y_data.flatten()):
    print("prediction: ", p, " true Y: ", y)
