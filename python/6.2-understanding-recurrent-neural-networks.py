#!/usr/bin/env python
# coding: utf-8

"""
Understanding recurrent neural networks 

This notebook contains the code samples found in Chapter 6, Section 2 of [Deep Learning with Python]
(https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).

SimpleRNN isn’t the only recurrent layer available in Keras. There are two others: LSTM and GRU. In 
practice, you’ll always use one of these, because SimpleRNN is generally too simplistic to be of real 
use. SimpleRNN has a major issue: although it should be able to retain at time t information about 
inputs seen many timesteps before, inpractice, such long-term dependencies are impossible to learn. 
This is due to the vanishing gradient problem, an effect that is similar to what is observed with 
non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers
to a network, the network eventually becomes untrainable. The theoretical reasons for this effect 
were studied by Hochreiter, Schmidhuber, and Bengio in the early 1990s. The LSTM and GRU layers are 
designed to solve this problem.

We consider the LSTM layer. The underlying Long Short-Term Memory (LSTM) algorithm was developed 
by Hochreiter and Schmidhuber in 1997;3 it was the culmination of their research on the vanishing 
gradient problem. This layer is a variant of the SimpleRNN layer you already know about; it adds a 
way to carry information across many timesteps. Imagine a conveyor belt running parallel to the 
sequence you’re processing. Information from the sequence can jump onto the conveyor belt at any 
point, be transported to a later timestep, and jump off, when you need it. This is essentially what 
LSTM does: it saves information for later, thus preventing older signals from gradually vanishing 
during processing. Because you’ll have a lot of weight matrices, index the W and U matrices in the 
cell with the letter o (Wo and Uo) for output.

For Major Snippets 

1.A recurrent layer in Keras

2.Preparing the IMDB data

3.Training the model with Embedding and SimpleRNN layers

4.A concrete LSTM example in Keras
"""


import keras
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 1.A recurrent layer in Keras

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()


# 2.Preparing the IMDB data

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# 3.Training the model with Embedding and SimpleRNN layers

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Plotting results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 4.A concrete LSTM example in Keras

# Using the LSTM layer in Keras

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Release the GPU memory
cuda.select_device(0)
cuda.close()