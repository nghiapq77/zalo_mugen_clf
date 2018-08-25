import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score

#from __future__ import absolute_import, division, print_function
import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch, genres
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel, createKerasModel, createKerasDataWithIdMusicFromSlices
from model import getKerasDataset

slicePath = "data/train_full/spectrograms/slices/"
nClasses = len(genres)


print("[+]Creating test data from slices")
test_x, test_y, idMusic = createKerasDataWithIdMusicFromSlices(slicePath, genres, sliceSize, validationRatio)
print("Create test data succesfully")
print("[+]Loading model")
model = createKerasModel(128,nClasses)
model.load_weights('keras_model.h5')
print("Load model successfully")
print("[+]Predicting")
pred  = model.predict(test_x)
print("Predict successfully")
preOrigin = pred
pred = pred.argmax(axis=1)
test_y = test_y.argmax(axis=1)
listId = []
testFix_y = []
index = 0;
for i in idMusic:
    flag = True
    for j in listId:
        if i==j:
            flag = False
            break
    if flag:
        listId.append(str(i))
        testFix_y.append(int(test_y[index]))
    index = index+1
testFix_y = np.asarray(testFix_y)
predictCount = np.zeros([len(listId),10])
index = 0
for i in idMusic:
    k = 0
    for j in listId:
        if i==j:
            predictCount[k,:] = predictCount[k,:] + preOrigin[index,:]
            break
        k = k+1;
    index = index+1
predictCount = predictCount.argmax(axis=1)
acc = predictCount==testFix_y
acc = 1*acc
print(np.mean(acc))
print(f1_score(predictCount, testFix_y, average='macro') )
print(f1_score(predictCount, testFix_y, average=None))


