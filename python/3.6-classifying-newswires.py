#!/usr/bin/env python
# coding: utf-8


import keras
keras.__version__


# Classifying newswires: a multi-class classification example

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# We have 8,982 training examples and 2,246 test examples:

len(train_data)
len(test_data)

# As with the IMDB reviews, each example is a list of integers (word indices):

train_data[10]

# Here's how you can decode it back to words, in case you are curious:

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Note that our indices were offset by 3 because 0, 1 and 2 are reserved indices for "padding", "start of sequence", 
# and "unknown".

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

decoded_newswire


# The label associated with an example is an integer between 0 and 45: a topic index.

train_labels[10]

# Preparing the data
# 
# We can vectorize the data with the exact same code as in our previous example:

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
 
# In our case, one-hot encoding of our labels consists in embedding each label as an all-zero vector with a 1 in 
# the place of the label index, e.g.:

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)


# Note that there is a built-in way to do this in Keras, which you have already seen in action in our MNIST example:

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# Building our network

# For this reason we will use larger layers. Let's go with 64 units:

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


# There are two other things you should note about this architecture:

# The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two 
# probability distributions: in our case, between the probability distribution output by our network, and the true 
# distribution of the labels. By minimizing the distance between these two distributions, we train our network to
# output something as close as possible to the true labels.

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Validating our approach

# Let's set apart 1,000 samples in our training data to use as a validation set:

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# Now let's train our network for 20 epochs:

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Let's display its loss and accuracy curves:

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure

"""
Traceback (most recent call last):
  File "3.6-classifying-newswires.py", line 153, in <module>
    acc = history.history['acc']
KeyError: 'acc'
"""

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# It seems that the network starts overfitting after 8 epochs. Let's train a new network from scratch for 8 epochs, 
# then let's evaluate it on the test set:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)


results

# Our approach reaches an accuracy of ~78%. With a balanced binary classification problem, the accuracy reached by a 
# purely random classifier would be 50%, but in our case it is closer to 19%, so our results seem pretty good, at least 
# when compared to a random baseline:


import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)


# Generating predictions on new data
# 
# We can verify that the `predict` method of our model instance returns a probability distribution over all 46 topics. 
# Let's generate topic predictions for all of the test data:

predictions = model.predict(x_test)

# Each entry in `predictions` is a vector of length 46:

predictions[0].shape

# The coefficients in this vector sum to 1:

np.sum(predictions[0])

# The largest entry is the predicted class, i.e. the class with the highest probability:

np.argmax(predictions[0])