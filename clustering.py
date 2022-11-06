#!/usr/bin/env python

'''

Clustering using the Iris data set < https://www.kaggle.com/datasets/uciml/iris > as example.

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

from sklearn.datasets import load_iris

# Load dataset locally or from the web.
data_raw = load_iris(as_frame = True)

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

# Show scatter matrix.
fig, ax = plt.subplots()
pd.plotting.scatter_matrix(data, ax = ax)
ax.set_title("Scatter plot of data.")
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

from sklearn.cluster import KMeans

model1 = KMeans(n_clusters = 3, random_state = 0)
model1.fit(data_train)
print("Labels after clustering with k-means:", model1.predict(data_train), "\n")

from sklearn.cluster import DBSCAN

model2 = DBSCAN(eps = 0.5, min_samples = 5, metric = "euclidean", n_jobs = -1)
model2.fit(data_train) # Note that these hyperparameters do no yield very good results ...

# DBSCAN has no predicz method. Thus, use another classifier.
from sklearn.neighbors import KNeighborsClassifier

clf1 = KNeighborsClassifier(n_neighbors = 10)
clf1.fit(data_train, model2.labels_)

print("Labels after clustering with DBSCAN and classification with KNN:", clf1.predict(data_train), "\n")

# ------------------------------------------------------------------------------
# Model tuning.

# Computing the inertias for different numbers of clusters.

# Loop over number of clusters.
inertia = []

for k in range(2, 11):
    model3 = KMeans(random_state = 0, n_clusters = k)
    model3.fit(data_train)
    inertia.append(model3.inertia_)
    print(f"Inertia for {k} clusters is {model3.inertia_}.")
print("\n")

# Plot inertia vs. number of clusters.
fig, ax = plt.subplots()
ax.plot(range(2, 11), inertia)
ax.set_title("Inertia vs. number of clusters for k-means.")
fig.show()

# Computing the silhouette scores for different numbers of clusters.
from sklearn.metrics import silhouette_score

# Loop over number of clusters.
silhouette = []

for k in range(2, 11):
    model4 = KMeans(random_state = 0, n_clusters = k)
    model4.fit(data_train)
    res = silhouette_score(data_train, model4.labels_)
    silhouette.append(res)
    print(f"Silhouette score for {k} clusters is {res}.")
print("\n")

# Plot silhouette score vs. number of clusters.
fig, ax = plt.subplots()
ax.plot(range(2, 11), silhouette)
ax.set_title("Silhouette score vs. number of clusters for k-means.")
fig.show()

# ------------------------------------------------------------------------------
# Apply PCA to reduce to two dimensions for plotting.

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(data_train)
print("Explained variances as result of PCA:", pca.explained_variance_ratio_)
print("Singular values as result of PCA:", pca.singular_values_)
data_train_reduced = pca.transform(data_train)

# Plot ground truth.
fig, ax = plt.subplots()
ax.scatter(data_train_reduced[:, 0], data_train_reduced[:, 1], c = data_raw["target"])
ax.set_title("Ground truth reduced to 2 dimensions via PCA.")
fig.show()

# Plot results of k-means.
fig, ax = plt.subplots()
ax.scatter(data_train_reduced[:, 0], data_train_reduced[:, 1], c = model1.predict(data_train))
ax.set_title("Clustering of k-means reduced to 2 dimensions via PCA.")
fig.show()

# Plot results of DBSCAN.
fig, ax = plt.subplots()
ax.scatter(data_train_reduced[:, 0], data_train_reduced[:, 1], c = clf1.predict(data_train))
ax.set_title("Clustering of DBSCAN reduced to 2 dimensions via PCA.")
fig.show()
