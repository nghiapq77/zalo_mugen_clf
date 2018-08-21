import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras

#from __future__ import absolute_import, division, print_function
import csv
import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch
from model import getDataset, createModel, createKerasModel, createKerasTestDataFromSlices
from model import getKerasDataset

slicePath = "data/test_full/spectrograms/slice/"

print("Loading model")
model = createKerasModel(128,10)
model.load_weights('keras_model.h5')
print("Load model successfully")

print("Creating test data from slices")
test_x, idMusic = createKerasTestDataFromSlices(slicePath, sliceSize, validationRatio)
print("Create test data succesfully")

print("Predicting ")
pred  = model.predict(test_x)
print("Predicting succesfully")
preOrigin = pred
pred = pred.argmax(axis=1)
listId = []
index = 0;
for i in idMusic:
    flag = True
    for j in listId:
        if i==j:
            flag = False
            break
    if flag:
        listId.append(str(i))
    index = index+1
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
result = [['Id',' Genre']]
result1 = [['Id',' Genre']]
index = 0;
for i in listId:
    result.append([ str(i)+'.mp3', str(predictCount[index])  ])
    result1.append([ str(i)+'.mp3', str(predictCount1[index]) ])
    index = index +1

with open("result.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(result)

with open("result1.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(result1)


