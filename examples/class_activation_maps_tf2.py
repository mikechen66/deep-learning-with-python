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
import tensorflow.keras.applications.inception_v3 as inception_v3
import tensorflow as tf
from numba import cuda


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Load the model
model = inception_v3.InceptionV3(weights='imagenet')

# Process the image 
img_name = '/home/mic/Documents/keras_examples/examples/meerkat.jpeg'
img = plt.imread(img_name)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()


# Process the image 
x = cv2.resize(img, (299, 299))
x = np.expand_dims(x, axis=0) 
x = inception_v3.preprocess_input(x)


# Predict the model 
preds = model.predict(x)
preds_decoded = inception_v3.decode_predictions(preds)[0]

_, label, conf = preds_decoded[0]
print("Label: %s with confidence %.2f%%" % (label, conf*100))


# OUTPUT: Label meerkat with confidence 93.51%

model.summary()


# Conduct the mixed10 layer 
last_conv_layer = model.get_layer('mixed10')

model_fm = tf.keras.Model(inputs=model.inputs,
                          outputs=[model.output,
                                   last_conv_layer.output])


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

print(CAM.shape)
plt.imshow(CAM)
plt.show()

# resize CAM
heatmap = cv2.resize(CAM, (img.shape[1], img.shape[0]))

# plot heatmap and image side by side
plt.subplots(1, 2, figsize=(15, 6))
plt.subplot(1,2,1)
plt.imshow(heatmap)
plt.title('Heatmap (resized CAM)')
plt.subplot(1,2,2)
plt.imshow(img)
plt.title('Original image')
plt.show()


# Plot the heatmap again 
plt.figure(figsize=(10, 10))
plt.imshow(img, alpha=0.5)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f'{label} with confidence {conf*100:.2f}%')
plt.axis('off')
#plt.savefig('./images/superimpose.png', dpi=300, bbox_inches='tight')
plt.show()


# Release the GPU memory
cuda.select_device(0)
cuda.close()