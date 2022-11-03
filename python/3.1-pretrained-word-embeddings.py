#!/usr/bin/env python
# coding: utf-8

# Text Similarity using Word Embeddings

# Input the path of program 
# $ cd ~/Documents/keras_examples

# Run the program 
# $ python 3.1-pretrained-word-embeddings.py


# In this notebook we're going to play around with pre build word embeddings and do some fun calculations:


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

# We'll start by downloading a pretrained model from GoogleNews. We're using `zcat` to unzip the file, so 
# you need to make sure you have that installed or replace it by something else.

# -model_path = "/home/mic/Documents/dataset/GoogleNews-vectors-negative300.bin"
MODEL = 'GoogleNews-vectors-negative300.bin'
path = get_file(MODEL + '.gz', 'https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/%s.gz' % MODEL)

if not os.path.isdir('generated'):
    os.mkdir('generated')

unzipped = os.path.join('generated', MODEL)
if not os.path.isfile(unzipped):
    with open(unzipped, 'wb') as fout:
        zcat = subprocess.Popen(['zcat'],
                          stdin=open(path),
                          stdout=fout
                         )
        zcat.wait()

# -model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)

# Let's take this model for a spin by looking at what things are most similar to espresso. 

model.most_similar(positive=['espresso'])

# Now for the famous equation, what is like woman if king is like man? We create a quick method to these calculations here:

def A_is_to_B_as_C_is_to(a, b, c, topn=1):
    a, b, c = map(lambda x:x if type(x) == list else [x], (a, b, c))
    res = model.most_similar(positive=b + c, negative=a, topn=topn)
    if len(res):
        if topn == 1:
            return res[0][0]
        return [x[0] for x in res]
    return None

A_is_to_B_as_C_is_to('man', 'woman', 'king')


# We can use this equation to acurately predict the capitals of countries by looking at what has the same relationship as 
# Berlin has to Germany for selected countries:

for country in 'Italy', 'France', 'India', 'China':
    print('%s is the capital of %s' % 
          (A_is_to_B_as_C_is_to('Germany', 'Berlin', country), country))

# Or we can do the same for important products for given companies. Here we seed the products equation with two products, 
# the iPhone for Apple and Starbucks_coffee for Starbucks. Note that numbers are replaced by # in the embedding model:

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

# And matplotlib to show the results. # As you can see, the countries, sports and drinks all form their own little clusters, 
# with arguably cricket and India attracting each other and maybe less clear, wine and France and Italy and espresso.

x=tsne[:,0]
y=tsne[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y)

for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0], (x1, y1), size=14)

plt.show()