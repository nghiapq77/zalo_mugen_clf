""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras

#from __future__ import absolute_import, division, print_function
import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel
from model import getKerasDataset

slicePath = "data/train_full/spectrograms/slices/"
genres = os.listdir(slicePath)
genres = [filename for filename in genres if os.path.isdir(slicePath+filename)]
nClasses = len(genres)

train_x, train_y, validation_x, validation_y = getKerasDataset(slicePath, genres, sliceSize, validationRatio)
print(train_y.shape)
model = keras.Sequential([
    keras.layers.Conv2D(
        input_shape=[128,128,1],
        filters=64,
        kernel_size=2,
        padding="same",
        activation="elu"
    ),
    keras.layers.MaxPooling2D(
        pool_size=2
    ),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=2,
        padding="same",
        activation="elu"
    ),
    keras.layers.MaxPooling2D(
        pool_size=2
    ),
    keras.layers.Conv2D(
        filters=256,
        kernel_size=2,
        padding="same",
        activation="elu"
    ),
    keras.layers.MaxPooling2D(
        pool_size=2
    ),
    keras.layers.Conv2D(
        filters=512,
        kernel_size=2,
        padding="same",
        activation="elu"
    ),
    keras.layers.MaxPooling2D(
        pool_size=2
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='elu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(nClasses, activation='softmax')
])

# Take a look at the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=nEpoch, validation_data=(validation_x, validation_y), batch_size=batchSize)
model.save('keras_model.h5')
"""
test_loss, test_acc = model.evaluate(validation_x, validation_y)

print('Validation accuracy:', test_acc)
"""