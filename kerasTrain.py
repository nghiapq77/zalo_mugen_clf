""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
#from __future__ import absolute_import, division, print_function
import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch, genres
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel, createKerasModel
from model import getKerasDataset
from sklearn.utils import class_weight
# from defineMetrics import Metrics 

slicePath = "data/train_full/spectrograms/slices/"
nClasses = len(genres)
train_x, train_y, validation_x, validation_y = getKerasDataset(slicePath, genres, sliceSize, validationRatio)
class_weights = train_y.argmax(axis=1)
class_weights = np.concatenate((np.asarray(range(10)),class_weights),axis=None)
class_weights = class_weight.compute_class_weight('balanced',np.unique(class_weights),class_weights)
class_weights = dict(enumerate(class_weights))
print(class_weights)
print(train_y.argmax(axis=1)) 
model = createKerasModel(128,nClasses)
# model.load_weights('keras_model.h5')

# Take a look at the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

model.fit(
    train_x, train_y,
    epochs=nEpoch,
    validation_data=(validation_x, validation_y),
    batch_size=batchSize,
    class_weight=class_weights
    )
model.save('keras_model.h5')
"""
test_loss, test_acc = model.evaluate(validation_x, validation_y)

print('Validation accuracy:', test_acc)
"""
