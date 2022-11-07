#!/usr/bin/env python
# coding: utf-8

"""
# Overfitting and underfitting
# 
# This notebook contains the code samples found in Chapter 3, Section 6 of [Deep Learning with 
# Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 
# 
# In all the examples we saw in the previous chapter -- movie review sentiment prediction, topic
# classification, and house price regression -- we could notice that the performance of our model
# on the held-out validation data would peak after a few epochs and would then start degrading, 
# i.e., the model would quickly start to overfit to the training data. It happens in every single 
# machine learning problem. Learning how to deal with overfitting is essential to mastering the 
# ML. 
# 
# The fundamental issue in the ML is the tension between optimization and generalization. The 
# former refers to the process of adjusting a model to get the best performance possible on 
# the training data (the learning in machine learning), and the latter refers to how well the 
# trained model performs on data it has never seen before. The goal is to get good generalization, 
# but you do not control generalization; you can only adjust the model based on its training data.
# 
# At the beginning of training, the two tasks are correlated: the lower your loss on training data, 
# the lower your loss on test data. While this is happening, your model is said to be under-fit: 
# there is still progress to be made; the network hasn't yet modeled all relevant patterns in the 
# training data. After a certain number of iterations on the training data, generalization stops 
# improving, validation metrics stall then start degrading: the model is then starting to overfit, 
# i.e., is it starting to learn patterns that are specific to the training data but misleading or 
# irrelevant when it comes to new data.
# 
# To prevent a model from learning misleading or irrelevant patterns found in the training data, 
# the best solution is to get more training data. A model trained on more data will generalize 
# better. When it is no longer possible, the next best solution is to modulate the quantity of 
# information that your model is allowed to store, or to add constraints on what information it 
# is allowed to store. If a network can only afford to memorize a small number of patterns, the 
# optimization process will force it to focus on the most prominent patterns, which have a better 
# chance of generalizing well.
# 
# The processing of fighting overfitting is called regularization. Please review some of the most 
# common regularization techniques, and apply them in practice to improve our movie classification 
# model from  the previous chapter.
"""


import keras
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    # Iterate the sequences in the enumerate()function that includes parameters: index and literal
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Fight overfitting

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])


# Now let's try to replace it with this smaller network:

smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

# Here's a comparison of the validation losses of the original network and the smaller network. The dots 
# are the validation loss values of the smaller network, and the crosses are the initial network (remember: 
# a lower validation loss signals a better model).

original_hist = original_model.fit(x_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))


smaller_model_hist = smaller_model.fit(x_train, y_train,
                                       epochs=20,
                                       batch_size=512,
                                       validation_data=(x_test, y_test))


epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']


# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# Build the model 

bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))


bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])


bigger_model_hist = bigger_model.fit(x_train, y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_test, y_test))


# Plot Epochs/Validation loss diagram...

bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# Plot training losses for our two networks:

original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()
plt.show()


# Add weight regularization to mitigate overfitting. 
l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])


# Estinate the impact of our L2 regularization penalty:
l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))

l2_model_val_loss = l2_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()


# L1 regularization
regularizers.l1(0.001)

# L1 and L2 regularization at the same time
regularizers.l1_l2(l1=0.001, l2=0.001)


# Add two Dropout layers in the IMDB network to esitmate reducing overfitting:
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))


dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])


dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))


dpt_model_val_loss = dpt_model_hist.history['val_loss']
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()


# Release the GPU memory
cuda.select_device(0)
cuda.close()