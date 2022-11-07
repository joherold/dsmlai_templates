#!/usr/bin/env python

'''

Multiclass classification using the MNIST data set < https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data > as example.

'''

import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import sklearn as skl
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------------------------
# Load data and do some exploration.

from sklearn.datasets import load_breast_cancer

# Load dataset locally or from the web.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ------------------------------------------------------------------------------
# Preprocessing of data.

# Flatten arrays.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# Scale data.
x_train = x_train / 255
x_test = x_test / 255

# ------------------------------------------------------------------------------
# Set up and train models.

from tensorflow.keras import layers

model1 = keras.models.Sequential([
    layers.InputLayer(input_shape = [784]),
    layers.Dense(300, activation = "relu"),
    layers.Dense(100, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])
model1.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
model1.summary()

history = model1.fit(x_train, y_train, epochs = 10, validation_split = 0.1)

# ------------------------------------------------------------------------------
# Plot training curves.

fig, ax = plt.subplots()
pd.DataFrame(history.history).plot(ax = ax)
ax.grid(True)
fig.show()

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

print("Performance on the test set:", model1.evaluate(x_test, y_test))
