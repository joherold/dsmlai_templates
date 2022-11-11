#!/usr/bin/env python

'''

Forecasting sequential data with RNNs and LSTMs using data generated via the Lotka-Volterra ODE as example.

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

# Generate data.
from scipy.integrate import ode

# Set number of time series to generate the data set from.
NM = 500

# Set time grid for evalution.
t0 = 0.0
t1 = 100.0
NTS = 1000
ts = np.linspace(t0, t1, NTS)
dt = (t1 - t0) / (NTS - 1)


# Set parameters for the model.
p = np.array([0.66, 1.33, 1, 1])

# Allocate memory and set initial values.
xs = np.zeros((NM, NTS, 2))
xs[:, 0, :] = [1.0, 1.0]

# Model RHS.
def f(t, x, p):

    # Allocate memory.
    f = np.zeros(2)

    f[0] = p[0] * x[0] - p[1] * x[0] * x[1]
    f[1] = p[2] * x[0] * x[1] - p[3] * x[1]

    return f

# Set integrator options.
r = ode(f)
r.set_integrator("vode", method = "bdf", rtol = 1e-06, atol = 1e-08)
r.set_f_params(p)

# Integrate NM times. These loops are inefficent.
for m in range(0, NM):

    r.set_initial_value(xs[m, 0, :], t0)

    for j in range(0, NTS - 1):

        r.integrate(r.t + dt)
        if r.successful() is not True:
            raise Exception
        else:
            xs[m, j + 1, :] = r.y

# Add some Gaussian noise if desired.
sigma = 0.25
xs_noise = xs + sigma * np.random.randn(xs.shape[0], xs.shape[1], xs.shape[2])

# Plot results.
fig, ax = plt.subplots()
ax.plot(ts, xs[0, :, 0], marker = "")
ax.plot(ts, xs[0, :, 1], marker = "")
ax.plot(ts, xs_noise[0, :, 0], marker = "o")
ax.plot(ts, xs_noise[0, :, 1], marker = "o")
ax.set_title("Exemplary solution of the ODE")
ax.legend(["x_0 = x_prey", "x_1 = x_predators", "x_0 with noise", "x_1 with noise"])
fig.show()

# Number of time steps to forecast.
NF = 100
NH = NTS - NF

# Split into training, valdiation, and test set. 70 %, 20 %, and 10%. Trying to forecast x_0 = x_prey for NF time steps.
b1 = int(NM * 0.7)
b2 = int(NM * 0.9)
x_train, y_train = xs_noise[:b1, :NH, 0], xs_noise[:b1, NH:, 0]
x_valid, y_valid = xs_noise[b1:b2, :NH, 0], xs_noise[b1:b2, NH:, 0]
x_test, y_test = xs_noise[b2:, :NH, 0], xs_noise[b2:, NH:, 0]

# ------------------------------------------------------------------------------
# Set up and train models.

# Build a simple RNN.
model1 = keras.models.Sequential([
    keras.layers.SimpleRNN(10, return_sequences = True, input_shape = [None, 1]),
    keras.layers.SimpleRNN(10),
    keras.layers.Dense(NF)
])

# Build a simple LSTM.
model2 = keras.models.Sequential([
    keras.layers.LSTM(10, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(10),
    keras.layers.Dense(NF)
])

# Print summaries.
model1.summary()
model2.summary()

# Train the models.
optimizer1 = keras.optimizers.Adam(learning_rate = 0.001)
model1.compile(loss = "mean_squared_error", optimizer = optimizer1)
model1.fit(x = x_train, y = y_train, epochs = 10, validation_data = (x_valid, y_valid))

optimizer2 = keras.optimizers.Adam(learning_rate = 0.001)
model2.compile(loss = "mean_squared_error", optimizer = optimizer1)
model2.fit(x = x_train, y = y_train, epochs = 10, validation_data = (x_valid, y_valid))

# ------------------------------------------------------------------------------
# Evaluate results on the test set.

print("Performance of the RNN on the test set:", model1.evaluate(x_test, y_test))
print("Performance of the LSTM on the test set:", model2.evaluate(x_test, y_test))
predictions1 = model1.predict(x_test)
predictions2 = model2.predict(x_test)

# Plot results for RNN.
fig, ax = plt.subplots()
ax.plot(ts[NH:], xs[0, NH:, 0], marker = "")
ax.plot(ts[NH:], xs[0, NH:, 1], marker = "")
ax.plot(ts[NH:], xs_noise[0, NH:, 0], marker = "o")
ax.plot(ts[NH:], xs_noise[0, NH:, 1], marker = "o")
ax.errorbar(ts[NH:], predictions1[0, :], marker = "x", markersize = 10)
ax.set_title("Prediction of the RNN for sample 0.")
ax.legend(["x_0 = x_prey", "x_1 = x_predators", "x_0 with noise", "x_1 with noise","predictions for x_0"])
fig.show()

# Plot results for LSTM network.
fig, ax = plt.subplots()
ax.plot(ts[NH:], xs[0, NH:, 0], marker = "")
ax.plot(ts[NH:], xs[0, NH:, 1], marker = "")
ax.plot(ts[NH:], xs_noise[0, NH:, 0], marker = "o")
ax.plot(ts[NH:], xs_noise[0, NH:, 1], marker = "o")
ax.errorbar(ts[NH:], predictions2[0, :], marker = "x", markersize = 10)
ax.set_title("Prediction of the LSTM for sample 0.")
ax.legend(["x_0 = x_prey", "x_1 = x_predators", "x_0 with noise", "x_1 with noise","predictions for x_0"])
fig.show()
