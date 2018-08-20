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
from model import getDataset, createModel, createKerasModel, createKerasTestDataFromSlices
from model import getKerasDataset

slicePath = "data/train_full/spectrograms/slices/"
genres = os.listdir(slicePath)
genres = [filename for filename in genres if os.path.isdir(slicePath+filename)]
nClasses = len(genres)


print("Creating test data from slices")
test_x, test_y, idMusic = createKerasTestDataFromSlices(slicePath, genres, sliceSize, validationRatio)
print("Create test data succesfully")
print("Loading model")
model = createKerasModel(128,nClasses)
model.load_weights('keras_model.h5')
print("Load model successfully")
pred  = model.predict(test_x)
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
predictCount1 = np.zeros([len(listId),10])
index = 0
for i in idMusic:
    k = 0
    for j in listId:
        if i==j:
            predictCount[k,pred[index]] = predictCount[k,pred[index]] + 1
            predictCount1[k,:] = predictCount1[k,:] + preOrigin[index,:]
            break
        k = k+1;
    index = index+1
predictCount = predictCount.argmax(axis=1)
predictCount1 = predictCount1.argmax(axis=1)
acc = predictCount==testFix_y
acc = 1*acc
acc1 = predictCount1==testFix_y
acc1 = 1*acc1
print(np.mean(acc))
print(np.mean(acc1))
print(1*(acc==acc1))
print(1*(predictCount1==predictCount))
print(testFix_y)


