#!/usr/bin/env python
# coding: utf-8

# Text generation with LSTM
# 
# The notebook contains the code samples found in Chapter 8, Section 1 of [Deep Learning with Python]
# (https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 
# 
# Let's put these ideas in practice in a Keras implementation. The first thing we need is a lot of text 
# data that we can use to learn a language model. You could use any sufficiently large text file or set 
# of text files -- Wikipedia, the Lord of the Rings, etc. In this example we use some of the writings 
# of Nietzsche, the late-19th century German philosopher (translated to English). The language model 
# we will learn will thus be specifically a model of Nietzsche's writing style and topics of choice, 
# rather than a more generic model of the English language.

# Given a trained model and a seed text snippet, we generate new text by repeatedly:
# 
# 1) Drawing from the model a probability distribution over the next character: 
# 2) Reweighting the distribution to a certain "temperature"
# 3) Sampling the next character at random according to the reweighted distribution
# 4) Adding the new character at the end of the available text


import keras
import tensorflow as tf 
import keras
import numpy as np
from keras import layers
import random
import sys
from numba import cuda


# Set up the GPU growth to avoid a sudden stop of the runtime with the reminding 
# message: Could not create cuDNN handle.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Preparing the data
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))


# Length of extracted character sequences
maxlen = 60

# We sample a new sequence every `step` characters
step = 3

# This holds our extracted sequences
sentences = []

# This holds the targets (the follow-up characters)
next_chars = []


for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))


# List of unique characters in the corpus
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)


# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Building the network
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))


# Since the targets are one-hot encoded, we will use categorical_crossentropy
# as the loss to train the model:
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# Training the language model and sampling from it
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for epoch in range(1, 60):
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# Release the GPU memory
cuda.select_device(0)
cuda.close()