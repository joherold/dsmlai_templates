#!/usr/bin/env python

'''

Anomaly detetion using the breast cancer Wisconsin data set < https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer > as example.

'''

import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import sklearn as skl
# import tensorflow as tf

# ------------------------------------------------------------------------------
# Load data and do some exploration.

from sklearn.datasets import load_breast_cancer

# Load dataset locally or from the web.
data_raw = load_breast_cancer(as_frame = True)

data = data_raw["data"]
data["target"] = data_raw["target"]

# Print some information on the dataframe, value counts, pivot tables, etc.
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(data.info(), "\n")
    print(data.describe(), "\n")

print("Print value counts for columns.\n")
for c in data.columns:
    print(data[c].value_counts(), "\n")

# Plot histograms.
fig, ax = plt.subplots()
data.hist(ax = ax)
ax.set_title("Histogram of data.")
fig.show()

# Show correlations on heatmap.
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax = ax)
ax.set_title("Correlation heatmap for numerical features.")
fig.show()

# Drop labels to simulate an unsupervised task.
data_train = data.drop(columns = ["target"])

# ------------------------------------------------------------------------------
# Preprocessing of data.

from sklearn.preprocessing import StandardScaler as Scaler
# from sklearn.preprocessing import MinMaxScaler as Scaler

# Scale data.
scaler1 = Scaler()
data_train[:] = scaler1.fit_transform(data_train)

# ------------------------------------------------------------------------------
# Set up and train models.

from sklearn.mixture import GaussianMixture

model1 = GaussianMixture(n_components = 1, random_state = 0)
model1.fit(data_train)

from sklearn.decomposition import PCA

model2 = PCA(n_components = 0.99, random_state = 0) # Retain 99% of variance.
model2.fit(data_train)

from sklearn.ensemble import IsolationForest

model3 = IsolationForest(random_state = 0)
model3.fit(data_train)

# ------------------------------------------------------------------------------
# Identify outliers via fitted models.

# Compute the q-the percentile for GM.
q1 = 2.0
densities = model1.score_samples(data_train)
threshold1 = np.percentile(densities, q1)
anomalies1 = data_train[densities < threshold1]
idx1 = anomalies1.index.to_numpy()
print(f"{anomalies1.shape[0]} anomalies identified via GM: \n", anomalies1)
print(f"Indices of the anomalies identified via GM: \n", np.sort(idx1), "\n")

# Compute the reconstruction error for PCA.
q2 = 98.0
data_train_reconstructed = model2.inverse_transform(model2.transform(data_train))
reconstruction_error = np.square(data_train - data_train_reconstructed).sum(axis = 1)
threshold2 = np.percentile(reconstruction_error, q2)
anomalies2 = data_train[reconstruction_error > threshold2]
idx2 = anomalies2.index.to_numpy()
print(f"{anomalies2.shape[0]} anomalies identified via PCA: \n", anomalies2)
print(f"Indices of the anomalies identified via PCA: \n", np.sort(idx2), "\n")

# Compute the anomaly scores for the isolation forest.
q3 = 2.0
anomaly_scores = model3.score_samples(data_train)
threshold3 = np.percentile(anomaly_scores, q3)
anomalies3 = data_train[anomaly_scores < threshold3]
idx3 = anomalies3.index.to_numpy()
print(f"{anomalies2.shape[0]} anomalies identified via isolation forest: \n", anomalies2)
print(f"Indices of the anomalies identified via isolation forest: \n", np.sort(idx3), "\n")
