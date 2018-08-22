""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras

#from __future__ import absolute_import, division, print_function
import os
import csv
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createKerasModel
from model import getKerasDataset

slicePath = "data/train_full/spectrograms/slices/"
csvfilepath = os.path.join("data/", "genres.csv")
genres = []
with open(csvfilepath, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        genres.append(int(rows[0]))
nClasses = len(genres)

train_x, train_y, validation_x, validation_y = getKerasDataset(slicePath, genres, sliceSize, validationRatio)

model = createKerasModel(128,nClasses)
model.load_weights('keras_model.h5')

# Take a look at the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=nEpoch, validation_data=(validation_x, validation_y), batch_size=batchSize)
model.save('keras_model2.h5')
"""
test_loss, test_acc = model.evaluate(validation_x, validation_y)

print('Validation accuracy:', test_acc)
"""