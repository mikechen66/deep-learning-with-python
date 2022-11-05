#!/usr/bin/env python
# coding: utf-8

# Text Similarity using Word Embeddings

# Input the path of program 
# $ cd ~/Documents/keras_examples

# Run the program 
# $ python lstm_stateful.py

# In this notebook we're going to play around with pre build word embeddings and do some fun 
# calculations:


import os
from keras.utils import get_file
import gensim
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(10, 10)

from sklearn.manifold import TSNE
import json
from collections import Counter
from itertools import chain

import tensorflow as tf
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# We'll start the program by the GoogleNews in the local file. 

model_path = "/home/mic/datasets/GoogleNews-vectors-negative300.bin"

# Local path for the model 
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Let's take this model for a spin by looking at what things are most similar to espresso.

# most_similar(): compute the cosine distance  
model.most_similar(positive=['espresso'])

# Now for the famous equation, what is like woman if king is like man? We create a quick 
# method to these calculations here:

def A_is_to_B_as_C_is_to(a, b, c, topn=1):
    # map() returns iterables, if no list disgnated, need to convert to the list format 
    # such as [x] that include x in the sqaure bracker [] in the else clause. 
    a, b, c = map(lambda x:x if type(x) == list else [x], (a, b, c))
    # topn: three words most similar 
    res = model.most_similar(positive=b + c, negative=a, topn=topn)
    if len(res):
        if topn == 1:
            return res[0][0]
        return [x[0] for x in res]
    return None

# For jupyter notebook or Terminal
# -A_is_to_B_as_C_is_to('man', 'woman', 'king')

# We can use this equation to acurately predict the capitals of countries by looking at what 
# has the same relationship as Berlin has to Germany for selected countries:

for country in 'Italy', 'France', 'India', 'China':
    print('%s is the capital of %s' % 
          (A_is_to_B_as_C_is_to('Germany', 'Berlin', country), country))
          # -(A_is_to_B_as_C_is_to('Germany', 'Berlin', positional placeholder), country))

# Or we can do the same for important products for given companies. Here we seed the products 
# equation with two products, the iPhone for Apple and Starbucks_coffee for Starbucks. Note 
# that numbers are replaced by # in the embedding model:

for company in 'Google', 'IBM', 'Boeing', 'Microsoft', 'Samsung':
    products = A_is_to_B_as_C_is_to(
        ['Starbucks', 'Apple'], 
        ['Starbucks_coffee', 'iPhone'], 
        company, topn=3)
    print('%s -> %s' % 
          (company, ', '.join(products)))

# Let's do some clustering by picking three categories of items, drinks, countries and sports:

beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
countries = ['Italy', 'Germany', 'Russia', 'France', 'USA', 'India']
sports = ['soccer', 'handball', 'hockey', 'cycling', 'basketball', 'cricket']

items = beverages + countries + sports
len(items)

# And looking up their vectors:

item_vectors = [(item, model[item]) 
                    for item in items
                    if item in model]
len(item_vectors)

# Now use TSNE for clustering:

vectors = np.asarray([x[1] for x in item_vectors])
lengths = np.linalg.norm(vectors, axis=1)
norm_vectors = (vectors.T / lengths).T

tsne = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(norm_vectors)

# And matplotlib to show the results. As you can see the countries, sports and drinks all form 
# their own little clusters, with arguably cricket and India attracting each other and maybe 
# less clear, wine and France and Italy and espresso.

x = tsne[:,0]
y = tsne[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y)

for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0], (x1, y1), size=14)

plt.show()


# Release the GPU memory
cuda.select_device(0)
cuda.close()