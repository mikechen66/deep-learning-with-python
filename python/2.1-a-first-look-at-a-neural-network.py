#!/usr/bin/env python
# coding: utf-8
# 
# A first look at a neural network
# 
# This notebook contains the code samples found in Chapter 2, Section 1 of [Deep Learning with Python]
# (https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).
# 
# We will now take a look at a first concrete example of a neural network, which makes use of the Python 
# library Keras to learn to classify hand-written digits. Unless you already have experience with Keras 
# or similar libraries, you will not understand everything about this first example right away.


import keras
keras.__version__
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist
from numba import cuda


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Let's have a look at the training data (only for Jupyter Notebook and Termninal Console)

# -train_images.shape
# -len(train_labels)
# -train_labels

# Let's have a look at the test data((only for Jupyter Notebook and Termninal Console))

# -test_images.shape
# -len(test_labels)
# -test_labels


# Build the network

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# Compile the network 

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# Train the images 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# We need to categorically encode the labels. 

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Fit the model to its training data.

network.fit(train_images, train_labels, epochs=5, batch_size=128)


# Performs well on the test set.

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# Release the GPU memory
cuda.select_device(0)
cuda.close()