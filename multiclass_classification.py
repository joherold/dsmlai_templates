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

# Reshape arrays.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Scale data.
x_train = x_train / 255
x_test = x_test / 255

# ------------------------------------------------------------------------------
# Set up and train models.

# Train an NN.
from tensorflow.keras import layers

model1 = keras.models.Sequential([
    layers.Flatten(input_shape = [28, 28, 1]),
    layers.Dense(units = 300, activation = "relu"),
    layers.Dense(units = 100, activation = "relu"),
    layers.Dense(units = 10, activation = "softmax")
])
model1.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
model1.summary()

# Measure wall time for training.
start_time = time.time()

# history1 = model1.fit(x_train, y_train, epochs = 10, validation_split = 0.1, verbose = 1)

print(f"Training of the NN took {time.time() - start_time} seconds. \n")

# Train a CNN. Make sure to have GPUs available.
model2 = keras.models.Sequential([
    layers.Conv2D(filters = 64, kernel_size = 7, activation = "relu", padding = "same", input_shape = [28, 28, 1]),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Conv2D(filters = 128, kernel_size = 3, activation = "relu", padding = "same"),
    layers.Conv2D(filters = 128, kernel_size = 3, activation = "relu", padding = "same"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Conv2D(filters = 256, kernel_size = 3, activation = "relu", padding = "same"),
    layers.Conv2D(filters = 256, kernel_size = 3, activation = "relu", padding = "same"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Flatten(),
    layers.Dense(units = 128, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(units = 64, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(units = 10, activation = "softmax")
])
model2.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model2.summary()

# Measure wall time for training.
start_time = time.time()

history2 = model2.fit(x_train, y_train, epochs = 10, validation_split = 0.1, verbose = 1)

print(f"Training of the CNN took {time.time() - start_time} seconds. \n")

# ------------------------------------------------------------------------------
# Plot training curves.

fig, ax = plt.subplots()
pd.DataFrame(history1.history).plot(ax = ax)
ax.grid(True)
ax.set_title("Training curves for the NN.")
fig.show()

fig, ax = plt.subplots()
pd.DataFrame(history2.history).plot(ax = ax)
ax.grid(True)
ax.set_title("Training curves for the CNN.")
fig.show()

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

print("Performance of the NN on the test set:", model1.evaluate(x_test, y_test))
print("Performance of the CNN on the test set:", model2.evaluate(x_test, y_test))
