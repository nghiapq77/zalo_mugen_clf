""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
""""""
import os
import errno
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image
from random import shuffle
import pickle
from config import validationRatio
from config import datasetPath

#processing input for train/test image
def getProcessedData(img, imgSize):
    imgData = np.asarray(img, dtype=np.uint8).reshape(imgSize, imgSize, 1)
    imgData = imgData/255
    return imgData
def getImageData(filename, imgSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imgSize)
    return imgData

#create datasets(train and valid) from slices
def createDatasetFromSlices(slicePath, genres, sliceSize, validationRatio):
    data = []
    for genre in genres:
        #get slices
        filenames = os.listdir(slicePath+genre)
        filenames = [filename for filename in filenames]
        #capped number of slices
        filenames = filenames[:2000] 
        #adding data
        for filename in filenames:
            imgData = getImageData(slicePath+genre+"/"+filename, sliceSize)
            label = [1. if genre == g else 0. for g in genres]
            data.append((imgData, label))

    shuffle(data)
    #splitting
    x,y = zip(*data)
    nValidation = int(len(x)*validationRatio)
    nTrain = len(x)-nValidation
    train_x = np.asarray(x[:nTrain]).reshape([-1, sliceSize, sliceSize, 1])
    train_y = np.asarray(y[:nTrain])
    validation_x = np.asarray(x[-nValidation:]).reshape([-1, sliceSize, sliceSize, 1])
    validation_y = np.asarray(y[-nValidation:])
    saveDataset(train_x, train_y, validation_x, validation_y, genres, sliceSize)
    return train_x, train_y, validation_x, validation_y

def getDatasetName(sliceSize):
    name = "{}sliceSize".format(sliceSize)
    return name

def saveDataset(train_x, train_y, validation_x, validation_y, genres, sliceSize):
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    datasetName = getDatasetName(sliceSize)
    pickle.dump(train_x, open("{}train_x_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_x, open("{}validation_x_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
    print("    Dataset saved!")

def loadDataset(genres, sliceSize):
    datasetName = getDatasetName(sliceSize)
    train_x = pickle.load(open("{}train_x_{}.p".format(datasetPath,datasetName), "rb" ))
    train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath,datasetName), "rb" ))
    validation_x = pickle.load(open("{}validation_x_{}.p".format(datasetPath,datasetName), "rb" ))
    validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
    print("    Training and validation datasets loaded! ")
    return train_x, train_y, validation_x, validation_y

#final func to create or load dataset 
def getDataset(slicePath, genres, sliceSize, validationRatio):
    print("[+] Dataset name: {}".format(getDatasetName(sliceSize)))
    if not os.path.isfile(datasetPath+"train_x_"+getDatasetName(sliceSize)+".p"):
        print("[+] Creating dataset with slices of size {}".format(sliceSize))
        createDatasetFromSlices(slicePath, genres, sliceSize, validationRatio) 
    else:
        print("[+] Using existing dataset")
    
    return loadDataset(genres, sliceSize)

#building model
def createModel(nbClasses,imageSize):
	print("[+] Creating model...")
	convnet = input_data(shape=[None, imageSize, imageSize, 1], name='input')

	convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)
	
	convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = fully_connected(convnet, 1024, activation='elu')
	convnet = dropout(convnet, 0.5)

	convnet = fully_connected(convnet, nbClasses, activation='softmax')
	convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

	model = tflearn.DNN(convnet)
	print("    Model created!")
	return model