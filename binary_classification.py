#!/usr/bin/env python

'''

Classification using the Titanic data set < https://www.openml.org/search?type=data&sort=runs&id=40945&status=active > as example.

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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load dataset locally or from the web.
data = fetch_openml("titanic", version = 1, as_frame = True)

y = data.target
x = data.data.drop(columns = ["boat", "body", "home.dest"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0, stratify = y)
data_train = x_train
data_train["survived"] = y_train

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

fig, ax = plt.subplots()
data_train.groupby("survived").hist(ax = ax)
ax.set_title("Histogram of data for survivors.")
fig.show()

pt = pd.pivot_table(data_train, values = "age", index = "survived", aggfunc = [np.median, np.std, np.min, np.max])
print("Pivot table: \n", pt, "\n")

# Show correlations on heatmap. Do this after transformations?
fig, ax = plt.subplots()
sns.heatmap(data_train.corr(), ax = ax)
ax.set_title("Correlation heatmap for numerical features.")
fig.show()

# ------------------------------------------------------------------------------
# Some exemplary feature engineering.

# Use only A,B,C,... to categorize cabins.
def categorize_cabins(passenger):
    if pd.isna(passenger["cabin"]):
        return None
    else:
        return passenger["cabin"][0]

x_train["cabin_categorized"] = x_train.apply(lambda row: categorize_cabins(row), axis = 1)
x_test["cabin_categorized"] = x_test.apply(lambda row: categorize_cabins(row), axis = 1)

# ------------------------------------------------------------------------------
# Preprocessing of data.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
imputer = SimpleImputer()
scaler = StandardScaler()

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "cabin_categorized"]
numerical_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked", "cabin_categorized"]

# Create one hot encodings for categorical values.
x_train = pd.get_dummies(x_train[features], drop_first = False)
x_test = pd.get_dummies(x_test[features], drop_first = False)
n_features = x_train.shape[1]

# Ensure same columns for train and test data after feature engineering.
x_test = x_test.reindex(columns = x_train.columns, fill_value = 0)

# Impute and scale data frames.
x_train_imputed = imputer.fit_transform(x_train)
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_imputed = imputer.transform(x_test)
x_test_scaled = scaler.transform(x_test_imputed)

# ------------------------------------------------------------------------------
# Set up and train models.

from sklearn.ensemble import RandomForestClassifier as Classifier1
from sklearn.ensemble import GradientBoostingClassifier as Classifier2

clf1 = Classifier1(n_estimators = 1000, max_depth = 100, random_state = 0)
clf1.fit(x_train_scaled, y_train)

y_train_pred = clf1.predict(x_train_scaled)
print("Confusion matrix: \n", confusion_matrix(y_train, y_train_pred), "\n")
print(y_train[y_train != y_train_pred], "\n")

clf2 = Classifier2(n_estimators = 1000, learning_rate = 1.0, max_depth = 100, random_state = 0)
clf2.fit(x_train_scaled, y_train)

# ------------------------------------------------------------------------------
# Cross validation.

from sklearn.model_selection import cross_val_score

print("Cross validation for random forest.")
scores = cross_val_score(clf1, x_train_scaled, y_train, cv = 10, scoring = "accuracy")
print("Scores for CV: \n", scores)
print("Median and STD: \n", np.median(scores), np.std(scores), "\n")

print("Cross validation for gradient boosting.")
scores2 = cross_val_score(clf2, x_train_scaled, y_train, cv = 10, scoring = "accuracy")
print("Scores for CV: \n", scores2)
print("Median and STD: \n", np.median(scores2), np.std(scores2), "\n")

# ------------------------------------------------------------------------------
# Model tuning via grid or random search.

from sklearn.model_selection import GridSearchCV

# Measure wall time.
start_time = time.time()

param_grid = [{"max_depth": [2, 4, 8, 16], "n_estimators": [100, 1000, 10000], "criterion": ["gini", "entropy", "log_loss"]}]
gsclf1 = GridSearchCV(estimator = clf1, param_grid = param_grid, n_jobs = -1, cv = 5)
gsclf1.fit(x_train_scaled, y_train)
print("Best estimator according to grid search: \n", gsclf1.best_estimator_)
print("... with score: \n", gsclf1.best_score_)

y_train_pred = gsclf1.predict(x_train_scaled)
print("Confusion matrix: \n", confusion_matrix(y_train, y_train_pred), "\n")
print(y_train[y_train != y_train_pred])

print(f"Grid search for random forest took {time.time() - start_time} seconds. \n")

# Measure wall time.
start_time = time.time()

param_grid = [{"max_depth": [2, 4, 8, 16], "n_estimators": [100, 1000, 10000]}]
gsclf2 = GridSearchCV(estimator = clf2, param_grid = param_grid, n_jobs = -1, cv = 5)
gsclf2.fit(x_train_scaled, y_train)
print("Best estimator according to grid search: \n", gsclf2.best_estimator_)
print("... with score: \n", gsclf2.best_score_)

print(f"Grid search for gradient boosting took {time.time() - start_time} seconds. \n")

# ------------------------------------------------------------------------------
# Build ensemble models.

from sklearn.ensemble import VotingClassifier

print("Build a voting classifier.\n")
vclf = VotingClassifier(estimators = [("clf1", clf1), ("clf2", clf2)], voting = "hard")
vclf.fit(x_train_scaled, y_train)

scores3 = cross_val_score(vclf, x_train_scaled, y_train, cv = 10, scoring = "accuracy")
print("Scores for CV: \n", scores3)
print("Median and STD: \n", np.median(scores3), np.std(scores3))

y_train_pred = vclf.predict(x_train_scaled)
print("Confusion matrix: \n", confusion_matrix(y_train, y_train_pred), "\n")
print(y_train[y_train != y_train_pred])

# ------------------------------------------------------------------------------
# Examine incorrectly classified passengers.

print("Examine incorrectly classified passengers ... \n")

err_predictions = x_train[y_train != y_train_pred]
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(err_predictions.info(), "\n")
    print(err_predictions.describe(), "\n")

for c in err_predictions.columns:
    print(err_predictions[c].value_counts(), "\n")

# ------------------------------------------------------------------------------
# Plot feature importances for random forest.

importances = clf1.feature_importances_
print("Feature importances obtained from random forest", importances, "\n")
idx = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(n_features), importances[idx])
ax.set_title("Importances by mean decrese in impurity.")
ax.set_yticks(range(n_features))
ax.set_yticklabels(np.array(x_train.columns)[idx])
fig.show()

# Plot feature importances with random pertubation.
from sklearn.inspection import permutation_importance

result = permutation_importance(clf1, x_train_scaled, y_train, n_repeats = 30, random_state = 0, n_jobs = -1)
importances = result.importances_mean
print("Feature importances obtained by random pertubation.", importances, "\n")
idx = np.argsort(importances)
n_features = len(importances)

fig, ax = plt.subplots()
ax.barh(range(n_features), importances[idx])
ax.set_title("Feature importances obtained by random pertubation.")
ax.set_yticks(range(n_features))
ax.set_yticklabels(np.array(x_train.columns)[idx])
fig.show()

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

from sklearn.metrics import accuracy_score
predictions = gsclf1.predict(x_test_scaled)
print("Final score:", accuracy_score(y_test, predictions))
