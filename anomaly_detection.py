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

model1 = GaussianMixture(n_components = 3, random_state = 0)
model1.fit(data_train)

# ------------------------------------------------------------------------------
# Compute the q-the percentile.

q = 1
densities = model1.score_samples(data_train)
threshold = np.percentile(densities, q)
anomalies = data_train[densities < threshold]
print(f"{anomalies.shape[0]} anomalies identified: \n", anomalies)
idx = anomalies.index.to_numpy()

# Plot the outliers.
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(data_train)
data_train_reduced = pca.transform(data_train)

fig, ax = plt.subplots()
ax.scatter(data_train_reduced[:, 0], data_train_reduced[:, 1], marker = "o")
ax.scatter(data_train_reduced[idx, 0], data_train_reduced[idx, 1], marker = "o")
ax.set_title("Data set reduced to 2 dimensions via PCA showing the computed anomalies.")
fig.show()
