#!/usr/bin/env python

'''

Text analysis with GRUs using the IMDB data set < https://keras.io/api/datasets/imdb/ > as example.

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
review_size = 200
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = vocab_size)

# Convert data sets to numpy arrays.
def imdb2array(x):

    x_array = np.zeros((x.shape[0], review_size))
    for i in range(0, x.shape[0]):
        j = 0
        while j < len(x[i]) and j < review_size:
            x_array[i, j] = x[i][j]
            j = j + 1

    return x_array

x_train = imdb2array(x_train)
x_test = imdb2array(x_test)

# Reconstruct reviews from the idx.
word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id + 3: word for word, id in word_index.items()}
id_to_word[0] = "<pad>"
id_to_word[1] = "<sos>"
id_to_word[2] = "<unk>"

def idx_to_words(idx):

    words = ""

    for id in idx:
        words = words + " " + id_to_word[int(id)]

    return words

# ------------------------------------------------------------------------------
# Set up and train models.

# Build a simple GRU.
embed_size = 64
num_oov_buckets = 100

model1 = keras.models.Sequential([
    keras.layers.Embedding(input_dim = vocab_size + num_oov_buckets, output_dim = embed_size, input_shape = [None]),
    keras.layers.GRU(32, return_sequences = True),
    keras.layers.GRU(32),
    keras.layers.Dense(1, activation = "sigmoid")
])

model1.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model1.summary()

# Measure wall time for training.
start_time = time.time()

history1 = model1.fit(x_train, y_train, epochs = 10, validation_split = 0.1, verbose = 1, batch_size = 128)

print(f"Training of the GRU took {time.time() - start_time} seconds. \n")

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

print("Performance of the GRU on the test set:", model1.evaluate(x_test, y_test), "\n")

# Compute the confusion matrix for the test set.
from sklearn.metrics import confusion_matrix

rounded_predictions = np.round(model1.predict(x_test))
print("Confusion matrix: \n", confusion_matrix(y_test, rounded_predictions), "\n")

# Print two examples from the test set.
print(idx_to_words(x_test[0]), "\nCategorized as:", rounded_predictions[0], "\nTrue label:", y_test[0], "\n")
print(idx_to_words(x_test[1]), "\nCategorized as:", rounded_predictions[1], "\nTrue label:", y_test[1], "\n")
