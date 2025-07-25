# -*- coding: utf-8 -*-
"""MNIST digit classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CmTQSLVd7BsWP-_7gVCR53t_e-bwO5Or
"""

pip install --upgrade keras

import keras
print(keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

x_train[0]

x_test[0]

y_train[0]

x_test[10]

y_test[10]

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

x_train.shape

import numpy as np

np.min(x_train[0])

np.max(x_train[0])

x_train=x_train.astype('float32')/255.0

x_test=x_test.astype('float32')/255.0

np.max(x_train[0])

np.min(x_train[0])

model=keras.Sequential()

model.add(keras.layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(28,28,1)
))

model.add(keras.layers.MaxPooling2D(
    pool_size=(2,2)
))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64,activation='relu'))

model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    verbose='auto'
)

model.evaluate(
    x=x_test,
    y=y_test,
    batch_size=32
)

model.predict(
    x_test,batch_size=32,verbose="auto"
)
