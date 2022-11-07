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
# on the held-out validation data would always peak after a few epochs and would then start degrading, 
# i.e. our model would quickly start to _overfit_ to the training data. Overfitting happens in every 
# single machine learning problem. Learning how to deal with overfitting is essential to mastering 
# machine learning.
# 
# The fundamental issue in machine learning is the tension between optimization and generalization. 
# Optimization refers to the process of adjusting a model to get the best performance possible on 
# the training data (the learning in machine learning), while generalization refers to how well
# the trained model would perform on data it has never seen before. The goal of the game is to get 
# good generalization, but you do not control generalization; you can only adjust the model based 
# on its training data.
# 
# At the beginning of training, optimization and generalization are correlated: the lower your loss on 
# training data, the lower your loss on test data. While this is happening, your model is said to be 
# under-fit: there is still progress to be made; the network hasn't yet modeled all relevant patterns 
# in the training data. But after a certain number of iterations on the training data, generalization 
# stops improving, validation metrics stall then start degrading: the model is then starting to overfit, 
# i.e., is it starting to learn patterns that are specific to the training data but misleading or 
# irrelevant when it comes to new data.
# 
# To prevent a model from learning misleading or irrelevant patterns found in the training data, the 
# best solution is to get more training data. A model trained on more data will naturally generalize 
# better. When that is no longer possible, the next best solution is to modulate the quantity
# of information that your model is allowed to store, or to add constraints on what information it is 
# allowed to store. If a network can only afford to memorize a small number of patterns, the optimization 
# process will force it to focus on the most prominent patterns, which have a better chance of generalizing 
# well.
# 
# The processing of fighting overfitting in this way is called regularization. Let's review some of the 
# most common regularization techniques, and let's apply them in practice to improve our movie classification 
# model from  the previous chapter.
"""


import keras
# -keras.__version__
from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

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
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Fighting overfitting

from keras import models
from keras import layers

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


import matplotlib.pyplot as plt

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# Plot the Epochs/Validation loss diagram...

# As you can see, the smaller network starts overfitting later than the reference one (after 6 epochs 
# rather than 4) and its performance degrades much more slowly once it starts overfitting.
# 
# Now, for kicks, let's add to this benchmark a network that has much more capacity, far more than the 
# problem would warrant:


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


# Here's how the bigger network fares compared to the reference one. The dots are the validation loss 
# values of the bigger network, and the crosses are the initial network.

bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# Plot Epochs/Validation loss diagram...


# The bigger network starts overfitting almost right away, after just one epoch, and overfits much more 
# severely. Its validation loss is also more noisy. Meanwhile, here are the training losses for our two networks:

original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()
plt.show()


# Adding weight regularization

from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

# Here's the impact of our L2 regularization penalty:

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


# Plot the Epochs/Validation loss diagram...

from keras import regularizers
from keras import models
from keras import layers
import numpy

# L1 regularization
regularizers.l1(0.001)

# L1 and L2 regularization at the same time
regularizers.l1_l2(l1=0.001, l2=0.001)


# For jupyter notebook 

# At training time: we drop out 50% of the units in the output

# -layer_output *= np.randint(0, high=2, size=layer_output.shape)

# At test time, we would be scaling the output down by the dropout rate. Here we scale by 0.5 (because we were 
# previous dropping half the units):

# At test time:
# -layer_output *= 0.5

# Note that this process can be implemented by doing both operations at training time and leaving the output 
# unchanged at test time, which is often the way it is implemented in practice:

# At training time:
# -layer_output *= np.randint(0, high=2, size=layer_output.shape)
# Note that we are scaling *up* rather scaling *down* in this case
# -layer_output /= 0.5


# -model.add(layers.Dropout(0.5))

# Let's add two `Dropout` layers in our IMDB network to see how well they do at reducing overfitting:

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