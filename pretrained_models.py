#!/usr/bin/env python

'''

Using pretrained models and transfer learning for classification with the flowers data set < https://www.tensorflow.org/datasets/catalog/tf_flowers > as example.

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

# Load dataset locally or from the web.
import tensorflow_datasets as tfds

(test_set, valid_set, train_set), info = tfds.load("tf_flowers",  split=["train[:70%]", "train[70%:85%]", "train[85%:]"], as_supervised = True, with_info = True)

# ------------------------------------------------------------------------------
# Preprocess data.

# Preprocess function to be passed to the data set. Resizes images to 224x224.
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.resnet.preprocess_input(resized_image)
    return final_image, label

# Add batch and prefetching to data set.
batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# ------------------------------------------------------------------------------
# Set up and train models.

# Use a pretrained ResNet50 model with weights for the ImageNet data set.
model1 = keras.applications.resnet.ResNet50(weights = "imagenet")
model1.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Print a summary of the model.
model1.summary()

# Check performance on the test set. Accuracy should be 0 %.
results1 = model1.evaluate(test_set)
print("Performance of the original ResNet50 on the test set:", results1)

# Remove the top layers.
from tensorflow.keras import layers

base_model2 = keras.applications.resnet.ResNet50(weights = "imagenet", include_top = False)

# Add a global averaging and a dense layer via the functional API.
n_classes = 5
avg = layers.GlobalAveragePooling2D()(base_model2.output)
output = layers.Dense(units = n_classes, activation = "softmax")(avg)
model2 = keras.Model(inputs = base_model2.input, outputs = output)

# Freeze the layers of the base model and retrain.
for layer in base_model2.layers:
    layer.trainable = False

# optimizer1 = keras.optimizers.SGD(learning_rate = 0.2, momentum = 0.9, decay = 0.01)
optimizer1 = keras.optimizers.Adam(learning_rate = 0.001)
model2.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer1, metrics = ["accuracy"])
history21 = model2.fit(train_set, epochs = 10, validation_data = valid_set)

# Unfreeze the layers of the base model and retrain.
for layer in base_model2.layers:
    layer.trainable = True

# optimizer2 = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, decay = 0.001)
optimizer2 = keras.optimizers.Adam(learning_rate = 0.0001)
model2.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer2, metrics = ["accuracy"])
history22 = model2.fit(train_set, epochs = 20, validation_data = valid_set)

# Check performance on the test set. Should be close to 85 %.
results2 = model2.evaluate(test_set)
print("Performance of the retrained ResNet50 on the test set:", results2)
