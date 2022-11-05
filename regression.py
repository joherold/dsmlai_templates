#!/usr/bin/env python

'''

Regression using the California housing prices data set < https://www.kaggle.com/datasets/camnugent/california-housing-prices > as example.

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

from sklearn.model_selection import train_test_split

# Load dataset locally or from the web.
data = pd.read_csv("housing.csv")

# Some feature engineering following A. Géron.
data["bedrooms_per_household"] = data["total_bedrooms"] / data["households"]
data["bedrooms_per_rooms"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

y = data["median_house_value"]
x = data.drop(columns = ["median_house_value"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) # Actually we should check if the sets needs to be stratified somehow ...

# Store training data into one data frame.
data_train = x_train
data_train["median_house_value"] = y_train

# Print some information on the dataframe, value counts, pivot tables, etc.
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(data_train.info(), "\n")
    print(data_train.describe(), "\n")

print("Print value counts for columns.\n")
for c in data_train.columns:
    print(data_train[c].value_counts(), "\n")

# Plot histograms and print pivot tables.
fig, ax = plt.subplots()
data_train.hist(ax = ax)
ax.set_title("Histogram of data.")
fig.show()

# Show correlations on heatmap.
fig, ax = plt.subplots()
sns.heatmap(data_train.corr(), ax = ax)
ax.set_title("Correlation heatmap for numerical features.")
fig.show()

# ------------------------------------------------------------------------------
# Preprocessing of data.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
imputer = SimpleImputer(strategy = "median")
scaler = StandardScaler()

# Select features to work with.
features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity", "bedrooms_per_household", "bedrooms_per_rooms", "population_per_household"]

# Create one hot encodings for categorical values.
x_train = pd.get_dummies(x_train[features], drop_first = True)
x_test = pd.get_dummies(x_test[features], drop_first = True)
n_features = x_train.shape[1]

# Ensure same columns for train and test data after introducing one hot encodings.
x_test = x_test.reindex(columns = x_train.columns, fill_value = 0)

# Impute and scale data frames.
x_train[:] = imputer.fit_transform(x_train)
x_train[:] = scaler.fit_transform(x_train)
x_test[:] = imputer.transform(x_test)
x_test[:] = scaler.transform(x_test)

# Plot histograms and print pivot tables again.
fig, ax = plt.subplots()
x_train.hist(ax = ax)
ax.set_title("Histogram of data.")
fig.show()

# Show correlations on heatmap again.
fig, ax = plt.subplots()
sns.heatmap(x_train.corr(), ax = ax)
ax.set_title("Correlation heatmap for numerical features.")
fig.show()

# ------------------------------------------------------------------------------
# Set up and train models.

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

model1 = Lasso(max_iter = 10000)
model1.fit(x_train, y_train)
score1 = model1.score(x_train, y_train)
print("Score for model1:", score1, "\n")

model2 = RandomForestRegressor(criterion = "squared_error", random_state = 0, n_jobs = -1)
model2.fit(x_train, y_train)
score2 = model2.score(x_train, y_train)
print("Score for model2:", score2, "\n")

# ------------------------------------------------------------------------------
# Cross validation.

from sklearn.model_selection import cross_val_score

print("Cross validation for Lasso.")
scores = cross_val_score(model1, x_train, y_train, cv = 5, n_jobs = -1)
print("Scores for CV: \n", scores)
print("Median and STD: \n", np.median(scores), np.std(scores), "\n")

print("Cross validation for random forest.")
scores2 = cross_val_score(model2, x_train, y_train, cv = 5, n_jobs = -1)
print("Scores for CV: \n", scores2)
print("Median and STD: \n", np.median(scores2), np.std(scores2), "\n")

# ------------------------------------------------------------------------------
# Model tuning via grid or random search.

from sklearn.model_selection import GridSearchCV

# Measure wall time.
start_time = time.time()

param_grid = [{"alpha": [1, 10, 100, 1000]}]
gsmodel1 = GridSearchCV(estimator = model1, param_grid = param_grid, n_jobs = -1, cv = 5)
gsmodel1.fit(x_train, y_train)
print("Best estimator according to grid search: \n", gsmodel1.best_estimator_)
print("... with score: \n", gsmodel1.best_score_)

print(f"Grid search for Lasso took {time.time() - start_time} seconds. \n")

# Measure wall time.
start_time = time.time()

param_grid = [{"max_depth": [None], "n_estimators": [100], "max_features": [0.3, 0.6, 0.9, 1], "max_samples": [0.3, 0.6, 0.9, 1]}]
gsmodel2 = GridSearchCV(estimator = model2, param_grid = param_grid, n_jobs = -1, cv = 5)
gsmodel2.fit(x_train, y_train)
print("Best estimator according to grid search: \n", gsmodel2.best_estimator_)
print("... with score: \n", gsmodel2.best_score_)

print(f"Grid search for random forest took {time.time() - start_time} seconds. \n")

# ------------------------------------------------------------------------------
# Plot feature importances.

# Plot feature importances with random pertubation.
from sklearn.inspection import permutation_importance

result = permutation_importance(gsmodel2, x_train, y_train, n_repeats = 30, random_state = 0, n_jobs = -1)
importances = result.importances_mean
print(importances)
idx = np.argsort(importances)
n_features = len(importances)

fig, ax = plt.subplots()
ax.barh(range(n_features), importances[idx])
ax.set_title("Importances by random pertubation.")
ax.set_yticks(range(n_features))
ax.set_yticklabels(np.array(x_train.columns)[idx])
fig.show()

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

from sklearn.metrics import mean_squared_error as metric
predictions = gsmodel2.predict(x_test)
print("Final score:", np.sqrt(metric(y_test, predictions)))
