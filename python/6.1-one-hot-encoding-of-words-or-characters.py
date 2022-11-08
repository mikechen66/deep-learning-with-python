#!/usr/bin/env python
# coding: utf-8

# One-hot encoding of words or characters

# This notebook contains the first code sample found in Chapter 6, Section 1 of [Deep Learning with Python]
# (https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 

# Note that Keras has built-in utilities for doing one-hot encoding text at the word level or character level, 
# starting from raw text data. This is what you should actually be using, as it will take care of a number of 
# important features, such as stripping special characters from strings, or only taking into the top N most 
# common words in your dataset (a common restriction to avoid dealing with very large input vector spaces).


import keras
import numpy as np
import string
from keras.preprocessing.text import Tokenizer


# 1.Word-level one-hot encoding (toy example)

# This is the initial data; one entry per sample
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Build an index of all tokens in the data.
token_index = {}
for sample in samples:
    # We simply tokenize the samples via the split method. In real life, we would also strip
    # punctuation and special characters from the samples.
    for word in sample.split():
        if word not in token_index: 
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            # Note that we don't attribute index 0 to anything.

# Vectorize the samples. We will only consider the first max_length words in each sample.
max_length = 10

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.


# 2.Character-level one-hot encoding (toy example)

# Character level one-hot encoding (toy example)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  # All printable ASCII characters.
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.

# Using Keras for word-level one-hot encoding:
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# We create a tokenizer, configured to only take into account the top-1000 most common words
tokenizer = Tokenizer(num_words=1000)
# This builds the word index
tokenizer.fit_on_texts(samples)

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(samples)

# You could also directly get the one-hot binary representations.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# This is how you can recover the word index that was computed
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Store our words as vectors of size 1000. Too many words will decrease the accuracy of this encoding method.
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # Hash the word into a "random" integer index
        # that is between 0 and 1000
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.