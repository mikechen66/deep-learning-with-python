#!/usr/bin/env python
# coding: utf-8

"""
Predicting house prices: a regression example

In the two previous examples, we considers classification problems, where the goal was to predict a 
single discrete label of an input data point. Another common type of machine learning problem is 
regression, which consists of predicting a continuous value instead of a discrete label. 

For instance, predicting the temperature tomorrow, given meteorological data, or predicting the time 
that a software project will take to complete, given its specifications.

Do not mix upn regression with the algorithm logistic regression: confusingly, logistic regression
is not a regression algorithm, it is a classification algorithm. 

The Boston Housing Price dataset

We will be attempting to predict the median price of homes in a given Boston suburb in the mid-1970s, 
given a few data points about the suburb at the time, such as the crime rate, the local property tax 
rate, etc.

The dataset has another interesting difference from our two previous examples: it has very few data 
points, only 506 in total, split between 404 training samples and 102 test samples, and each feature
in the input data (e.g. the crime rate is a feature) has a different scale. For instance some values 
are proportions, which take a values between 0 and 1, others take values between 1 and 12, others 
between 0 and 100...


Training and test samples

# We have 404 training samples and 102 test samples. The data comprises 13 features. The 13 features 
# in the input data are as follow:
# 
# 1. Per capita crime rate.
# 2. Proportion of residential land zoned for lots over 25,000 square feet.
# 3. Proportion of non-retail business acres per town.
# 4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 5. Nitric oxides concentration (parts per 10 million).
# 6. Average number of rooms per dwelling.
# 7. Proportion of owner-occupied units built prior to 1940.
# 8. Weighted distances to five Boston employment centres.
# 9. Index of accessibility to radial highways.
# 10. Full-value property-tax rate per $10,000.
# 11. Pupil-teacher ratio by town.
# 12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# 13. % lower status of the population.
"""


import keras
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Predicting house prices: a regression example
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()


# Only for the corresponding Jupyter Notebook 

# -train_data.shape
# (404, 13)

# -test_data.shape
# (102, 13)

# The targets are the median values of owner-occupied homes, in thousands of dollars:
# -train_targets
"""
array([ 15.2, 42.3, 50. , 21.1, 17.7, 18.5, 11.3, 15.6, 15.6, 
        14.4, 12.1, 17.9, 23.1, 19.9, 15.7, 8.8, 50. , 22.5,
        ....................................................
        26.5, 28.7, 13.3, 10.4, 24.4, 23. , 20. , 17.8, 7. , 
        11.8, 24.4, 13.8, 19.4, 25.2, 19.4, 19.4, 29.1])
"""


# Preparing the data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# Building the network


def build_model():
    # Because we will need to instantiate the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)

    # Build the Keras model(already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


# Only for the corresponding Jupyter Notebook 

# -all_scores
# [1.9374959468841553, 2.1278462409973145, 2.410356283187866, 2.3951032161712646]

# -np.mean(all_scores)
# 2.21770042181015


# Some memory clean-up
K.clear_session()

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)

    # Build the Keras model(already compiled)
    model = build_model()
    # Train the model(in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# Plot the diagram:
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# Plot Epochs/Validation MAE again....

# Get a fresh and compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# Only for the corresponding Jupyter Notebook 
# -test_mae_score
# -2.632322072982788


# Release the GPU memory
cuda.select_device(0)
cuda.close()