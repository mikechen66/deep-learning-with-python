#!/usr/bin/env python
# coding: utf-8

import keras
keras.__version__


# The IMDB dataset

# The following code will load the dataset (when you run it for the first time, about 80MB of data 
# will be downloaded to your machine):

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# decoded_review

# We will go with the latter solution. Let's vectorize our data, which we will do manually for maximum 
# clarity:

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)


# We should also vectorize our labels, which is straightforward:

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Now our data is ready to be fed into a neural network.

# Building our network

# Here's what our network looks like:
# 
# [3-layer network](https://s3.amazonaws.com/book.keras.io/img/ch3/3_layer_network.png)

# And here's the Keras implementation, very similar to the MNIST example you saw previously:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Here's the step where we configure our model with the `rmsprop` optimizer and the `binary_crossentropy` 
# loss function. Note that we will also monitor accuracy during training.

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# We are passing our optimizer, loss function and metrics as strings, which is possible because `rmsprop`, 
# `binary_crossentropy` and `accuracy` are packaged as part of Keras. Sometimes you may want to configure 
# the parameters of your optimizer, or pass a custom loss function or metric function. This former can be 
# done by passing an optimizer class instance as the `optimizer` argument:


from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# The latter can be done by passing function objects as the `loss` or `metrics` arguments:


from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# Validating our approach

# In order to monitor during training the accuracy of the model on data that it has never seen before, 
# we will create a "validation set" by setting apart 10,000 samples from the original training data:

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# We will now train our model for 20 epochs (20 iterations over all samples in the `x_train` and `y_train` 
# tensors), in mini-batches of 512 samples. At this same time we will monitor loss and accuracy on the 10,000 
# samples that we set apart. This is done by passing the validation data as the `validation_data` argument:

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# Note that the call to `model.fit()` returns a `History` object. This object has a member `history`, which is 
# a dictionary containing data about everything that happened during training. Let's take a look at it:

"""
history_dict = history.history
history_dict.keys()
dict_keys(['val_acc', 'acc', 'val_loss', 'loss']) 
"""

# It contains 4 entries: one per metric that was being monitored, during training and during validation. Let's 
# use Matplotlib to plot the training and validation loss side by side, as well as the training and validation 
# accuracy:

import matplotlib.pyplot as plt

"""
Traceback (most recent call last):
  File "3.5-classifying-movie-reviews_no_comments.py", line 141, in <module>
    val_acc = history.history['acc']
KeyError: 'acc'
"""

val_acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Let's train a new network from scratch for four epochs, then evaluate it on our test data:

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

"""
>>>results 
[0.2929924130630493, 0.88327999999999995]
""" 

"""

model.predict(x_test)
>>> model.predict(x_test) array([[ 0.98006207]
[ 0.99758697]
[ 0.99975556]
...,
[ 0.82167041]
[ 0.02885115]
[ 0.65371346]], dtype=float32)
""" 