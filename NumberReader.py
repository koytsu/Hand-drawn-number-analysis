#always use sparce
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28,)))
model.add(keras.layers.Dense(748, activation='tanh'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
	loss=keras.losses.sparse_categorical_crossentropy,
	metrics=[keras.metrics.sparse_categorical_accuracy])
(x_train, y_train), (x_test, y_test) = mnist.load_data()


model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_data=(x_test, y_test))



