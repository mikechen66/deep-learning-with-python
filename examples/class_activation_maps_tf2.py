"""
Class activation Maps 

Class activation Maps (shortly CAM) was introduced in the paper Learning Deep Features for Discriminative 
Localization by Zhou et al(2016). It can help explain the predictions of a ConvNet. The authors also use 
this method to localize objects without providing the model with any bounding box annotations. The model 
just learns the classification task with class labels and is then able to localize the object of a specific 
class in an image. 

Learning Deep Features for Discriminative Localization

https://arxiv.org/abs/1512.04150

Breif Introduction

https://www.pinecone.io/learn/class-activation-maps/

"""


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# pretrained inception model
import tensorflow.keras.applications.inception_v3 as inception_v3

import tensorflow as tf
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# load model
model = inception_v3.InceptionV3(weights='imagenet')

# - IMG_NAME = 'meerkat.jpg'
img_name = '/home/mic/Documents/keras_examples/examples/meerkat.jpeg'
img = plt.imread(img_name)

x = cv2.resize(img, (299, 299))
x = np.expand_dims(x, axis=0) 
x = inception_v3.preprocess_input(x)

preds = model.predict(x)

preds_decoded = inception_v3.decode_predictions(preds)[0]

_, label, conf = preds_decoded[0]
print("Label: %s with confidence %.2f%%" % (label, conf*100))

# OUTPUT: Label meerkat with confidence 93.51%

model.summary()

last_conv_layer = model.get_layer('mixed10')

model_fm = tf.keras.Model(inputs=model.inputs,
                          outputs=[
                              model.output,
                              last_conv_layer.output
                          ])

model_out, feature_maps = model_fm.predict(x)

# get rid of the batch channel, e.g. (1, 1000) -> (1000,)
feature_maps = np.squeeze(feature_maps)
model_out = np.squeeze(model_out)

print(model_out.shape)        # (1000,)
print(feature_maps.shape)     # (8, 8, 2048)


# get weights of last layer
weights = model.layers[-1].weights[0]
print(weights.shape)       # (2048, 1000)

# find winning class (highest confidence)
max_idx = np.argmax( model_out )
print(f"Max index: {max_idx} ({model_out[max_idx]*100:.2f}%)")
# OUTPUT: Max index = 299 (93.51%)

winning_weights = weights[:, max_idx]
print(winning_weights.shape)    # (2048,)

CAM = np.sum(feature_maps * winning_weights, axis=2)

# resize CAM
heatmap = cv2.resize(CAM, (img.shape[1], img.shape[0]))


# Release the GPU memory
cuda.select_device(0)
cuda.close()