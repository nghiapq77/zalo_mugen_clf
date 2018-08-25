""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
""""""
import os
import errno
import numpy as np
import tflearn
import tensorflow as tf
from tensorflow import keras
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image
from random import shuffle
import pickle
from config import validationRatio
from config import datasetPath
import scipy.io as scipy

def saveAsMat(matrix,name):
    print("Saving " + name)
    pathName = '/Users/mpxt2/'+name +'.mat'
    scipy.savemat(pathName, mdict={name: matrix})
    print("Saving " + name  + " sucssfully")

def convertOutputToGenre(output):
    return np.argmax(output,axis=1)

#processing input for train/test image
def getProcessedData(img, imgSize):
    imgData = np.asarray(img, dtype=np.uint8).reshape(imgSize, imgSize, 1)
    imgData = imgData/255.
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
        filenames = filenames[:1000] 
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
###############3
"""datasetforkeras"""
def createKerasDatasetFromSlices(slicePath, genres, sliceSize, validationRatio):
    print("[+] Creating dataset")
    data = []  
    #get slices
    filenames = os.listdir(slicePath)
    filenames = [filename for filename in filenames]
    shuffle(filenames)
    #capped number of slices
    cappedSlices = 50000
    filenames = filenames[:cappedSlices] 
    #adding data
    for filename in filenames:
        img = Image.open(slicePath+"/"+filename)
        imgData = np.asarray(img, dtype=np.uint8)
        imgData = imgData/255.
        genre = filename.split('_')[0]
        label = [1. if genre == g else 0. for g in genres]
        data.append((imgData, label))

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
######################

def createKerasDataWithIdMusicFromSlices(slicePath, genres, sliceSize, validationRatio):
    data = []
    #get slices
    filenames = os.listdir(slicePath)
    filenames = [filename for filename in filenames]
    #capped number of slices
    cappedSlices = 5000
    shuffle(filenames)
    filenames = filenames[:cappedSlices] 
    #adding data
    for filename in filenames:
        img = Image.open(slicePath+"/"+filename)
        imgData = np.asarray(img, dtype=np.uint8)
        imgData = imgData/255.
        genre = filename.split('_')[0]
        label = [1. if genre == g else 0. for g in genres]
        data.append((imgData, label,filename.split('_')[1]))
    #splitting
    test_x,test_y,idMusic = zip(*data)
    return np.asarray(test_x).reshape([-1, sliceSize, sliceSize, 1]), np.array(test_y), np.array(idMusic)

def createKerasTestDataFromSlices(slicePath, sliceSize, validationRatio):
    data = []
    filenames = os.listdir(slicePath)
    filenames = [filename for filename in filenames]
    for filename in filenames:
        img = Image.open(slicePath+"/"+filename)
        imgData = np.asarray(img, dtype=np.uint8)
        imgData = imgData/255.
        data.append((imgData,filename.split('_')[0]))
    test_x,idMusic = zip(*data)
    return np.asarray(test_x).reshape([-1, sliceSize, sliceSize, 1]), np.array(idMusic)

def getDatasetName(sliceSize):
    name = "{}sliceSize".format(sliceSize)
    return name

def saveDataset(train_x, train_y, validation_x, validation_y, genres, sliceSize):
    print("[+] Saving dataset")
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
    print('[+]Loading data')
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
        return createDatasetFromSlices(slicePath, genres, sliceSize, validationRatio) 
    else:
        print("[+] Using existing dataset")
    
    return loadDataset(genres, sliceSize)
def getKerasDataset(slicePath, genres, sliceSize, validationRatio):
    print("[+] Dataset name: {}".format(getDatasetName(sliceSize)))
    if not os.path.isfile(datasetPath+"train_x_"+getDatasetName(sliceSize)+".p"):
        print("[+] Creating dataset with slices of size {}".format(sliceSize))
        return createKerasDatasetFromSlices(slicePath, genres, sliceSize, validationRatio) 
    else:
        print("[+] Using existing dataset")
    
    return loadDataset(genres, sliceSize)

def createKerasModel(imageSize,nClasses):
    model =   keras.Sequential([
        keras.layers.Conv2D(
            input_shape=[imageSize,imageSize,1],
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
        keras.layers.Dense(128, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(nClasses, activation='softmax')
    ])
    return model
    
#building model
def createModel(nbClasses,imageSize,learningRate):
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

	convnet = fully_connected(convnet, 1028, activation='elu')
	convnet = dropout(convnet, 0.5)

	convnet = fully_connected(convnet, nbClasses, activation='softmax')
	convnet = regression(convnet, learning_rate= learningRate, optimizer='rmsprop', loss='categorical_crossentropy')

	model = tflearn.DNN(convnet)
	print("    Model created!")
	return model