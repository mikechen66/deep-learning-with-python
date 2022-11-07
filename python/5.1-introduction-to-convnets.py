#!/usr/bin/env python
# coding: utf-8

# 5.1 - Introduction to convnets
# 
# This code contains the code sample found in Chapter 5, Section 1 of [Deep Learning with Python]
# (https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).
# 
# First, let's take a practical look at a very simple convnet example. We will use our convnet to 
# classify MNIST digits, a task that you've already been through in Chapter 2, using a densely 
# connected network (our test accuracy then was 97.8%). Even though our convnet will be very basic, 
# its accuracy will still blow out of the water that of the densely-connected model from Chapter 2.
# 
# The 6 lines of code below show you what a basic convnet looks like. It's a stack of Conv2D and 
# MaxPooling2D layers. We'll see in a minute what they do concretely.

# Importantly, a convnet takes as input tensors of shape (image_height, image_width, image_channels) 
# (not including the batch dimension). In our case, we will configure our convnet to process inputs 
# of size (28, 28, 1), which is the format of MNIST images. We do this via passing the argument 
# input_shape=(28, 28, 1) to our first layer.


import keras
keras.__version__

import tensorflow as tf 
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from numba import cuda


# Set up the GPU growth to avoid a sudden stop of the runtime with the reminding 
# message: Could not create cuDNN handle.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Initiate the model 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Display the architecture of our convnet

model.summary()

# You can see above that the output of every `Conv2D` and `MaxPooling2D` layer is a 3D tensor of shape 
# (height, width, channels). The width and height dimensions tend to shrink as we go deeper in the 
# network. The number of channels is controlled by the first argument passed to the Conv2D layers 
# (e.g. 32 or 64).
# 
# The next step would be to feed our last output tensor (of shape (3, 3, 64)) into a densely-connected 
# classifier network like those you are familiar with: a stack of Dense layers. The classifiers 
# process vectors, which are 1D, whereas our current output is a 3D tensor. So first, we will have to 
# flatten our 3D outputs to 1D, and then add a few Dense layers on top:

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# We are going to do 10-way classification, so we use a final layer with 10 outputs and softmax. Here's 
# what our network looks like:

model.summary()


# As you can see, our (3, 3, 64) outputs were flattened into vectors of shape (576,), before going through
# two Dense layers.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Let's evaluate the model on the test data:

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc


# Release the GPU memory
cuda.select_device(0)
cuda.close()