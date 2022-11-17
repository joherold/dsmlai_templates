#!/usr/bin/env python

'''

Representation learning with autoencoders using the Fashion MNIST data set < https://keras.io/api/datasets/fashion_mnist/ > as example.

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

# Load data set locally or from the web.
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Scale data.
x_train = x_train / 255
x_test = x_test / 255

# Split triaining set.
n_validation = 10000
x_valid = x_train[:-n_validation]
y_valid = y_train[:-n_validation]
x_train = x_train[-n_validation:]
y_train = y_train[-n_validation:]

# ------------------------------------------------------------------------------
# Set up and train models.

# Build an autoencoder.
encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(100, activation = "selu"),
    keras.layers.Dense(30, activation = "selu")
])
decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = "selu", input_shape = [30]),
    keras.layers.Dense(28 * 28, activation = "sigmoid"),
    keras.layers.Reshape([28, 28]),
])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss = "binary_crossentropy", optimizer = "adam")
encoder.summary()
decoder.summary()
autoencoder.summary()

# Measure wall time for training.
start_time = time.time()

history1 = autoencoder.fit(x = x_train, y = x_train, validation_data = [x_valid, x_valid], epochs = 10, verbose = 1)

print(f"Training of the autoencoder took {time.time() - start_time} seconds. \n")

# ------------------------------------------------------------------------------
# Evaluate results by plotting some reconstructions.
n_images = 10
reconstructions = autoencoder.predict(x_test[:n_images])

fig, axs = plt.subplots(2, n_images)
for i in range(0, n_images):
    axs[0][i].imshow(x_test[i], cmap = "binary")
    axs[1][i].imshow(reconstructions[i], cmap = "binary")
fig.show()

# Visualize the test set via t-SNE.
from sklearn.manifold import TSNE

reconstructions = encoder.predict(x_test)
tsne = TSNE()
reconstructions_2d = tsne.fit_transform(reconstructions)

fig, ax = plt.subplots()
ax.scatter(reconstructions_2d[:, 0], reconstructions_2d[:, 1], c = y_test)
fig.show()
