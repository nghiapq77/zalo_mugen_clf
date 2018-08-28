from subprocess import Popen, PIPE, STDOUT
from random import shuffle

import gc
import os
import csv
import numpy as np 
import librosa as lbr

from config import genres
from configMode2 import trainCsvPath, audioPath, slicePath, datasetPath
from configMode2 import pixelPerSecond, sliceSize, testRatio, validationRatio
from configMode2 import nameTrain_x, nameTrain_y, nameVal_x, nameVal_y, nameTest_x, nameTest_y, nameTest_id

WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

########################### Process Mp3  ###############################
########################################################################
currentPath = os.path.dirname(os.path.realpath(__file__))

def createFeaturesFromAudio(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

    features[features == 0] = 1e-6
    return (np.log(features), float(new_input.shape[0]) / sample_rate)

def sliceAudio(filename):
    tmp, _ = createFeaturesFromAudio(filename)
    nSamples = int(tmp.shape[0]/sliceSize)
    slices = []
    for i in range(nSamples):
        start = i*sliceSize
        slice = tmp[start:start+sliceSize][:sliceSize]
        slices.append(slice)
    return slices

def createSlicesFromAudio():
    #creating folders
    if not os.path.exists(slicePath):
        os.mkdir(slicePath)
    for g in genres:
        path = os.path.join(slicePath, "{}".format(g))
        if not os.path.exists(path):
            os.mkdir(path)
    with open(trainCsvPath, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            genre = rows[1]
            name = rows[0]
            audioFilePath = os.path.join(audioPath, name)
            if os.path.exists(audioFilePath):
                slices = sliceAudio(audioFilePath)
                label = [1. if genre == g else 0. for g in genres]
                i = 0
                for slice_i in slices:
                    data = (slice_i, label)
                    outfile = os.path.join(slicePath, "{}/{}_{}.npy".format(genre, name[:-4], i))
                    np.save(outfile, data)
                    i+=1
                print("Finished processing file {}".format(name))

########################### Process slice ##############################
########################################################################

def createDataset():
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)
    data = []
    for genre in genres:
        filenames = os.listdir(slicePath+genre)
        filenames = [filename for filename in filenames]
        for filename in filenames:
            if not filename.endswith(".npy"):
                continue
            infile = os.path.join(slicePath, "{}/{}".format(genre, filename))
            idMusic = filename.split('_')[0]
            npData = np.load(infile)
            newData = tuple(npData)+(idMusic,)
            data.append(np.asarray(newData))

    #################Splitting##############
    shuffle(data)
    x,y,idMusic = zip(*data)
    data = None
    gc.collect() ## release memory 
    nValidation = int(len(x)*validationRatio)
    nTest = int(len(x)*testRatio)
    nTrain = len(x)-nValidation - nTest
    test_id = np.asarray(idMusic[-nTest:])
    idMusic = None
    gc.collect()

    # split _x 
    train_x = np.asarray(x[:nTrain]).reshape([-1, sliceSize, sliceSize, 1])
    val_x = np.asarray(x[-nValidation-nTest:]).reshape([-1, sliceSize, sliceSize, 1])
    x = None  ## release memory
    gc.collect()
    test_x = val_x[-nTest:]
    val_x = val_x[:nValidation]

    # normalizing features
    train_x = lbr.util.normalize(train_x, norm=2)
    val_x = lbr.util.normalize(val_x, norm=2)
    test_x = lbr.util.normalize(test_x, norm=2)

    # split _y
    train_y = np.asarray(y[:nTrain])
    val_y = np.asarray(y[-nValidation-nTest:])
    y = None  ## release memory
    gc.collect()
    test_y = val_y[-nTest:]
    val_y = val_y[:nValidation]

    #saving
    print("[+] Saving dataset")
    outfile = os.path.join(datasetPath, nameTrain_x)
    np.save(outfile,train_x)
    outfile = os.path.join(datasetPath, nameTrain_y)
    np.save(outfile,train_y)
    outfile = os.path.join(datasetPath, nameVal_x)
    np.save(outfile,val_x)
    outfile = os.path.join(datasetPath, nameVal_y)
    np.save(outfile,val_y)
    outfile = os.path.join(datasetPath, nameTest_x)
    np.save(outfile,test_x)
    outfile = os.path.join(datasetPath, nameTest_y)
    np.save(outfile,test_y)
    outfile = os.path.join(datasetPath, nameTest_id)
    np.save(outfile,test_id)
    return train_x, train_y, val_x, val_y, test_x, test_y, test_id

########################### Process dataset ############################
########################################################################

def checkExistDataset():
    dataExist = True
    if not os.path.isfile(datasetPath+nameTrain_x):
        print("\t train_x data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameTrain_y):
        print("\t train_y data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameVal_x):
        print("\t val_x data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameVal_y):
        print("\t val_y data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameTest_x):
        print("\t test_x data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameTest_y):
        print("\t test_y data is not exist")
        dataExist = False
    if not os.path.isfile(datasetPath+nameTest_id):
        print("\t test_id data is not exist")
        dataExist = False
    return dataExist

def getDataset():
    if not checkExistDataset() :
        print("[+] Creating new dataset")
        return createDataset() 
    else:
        print("[+] Loading exist dataset")
    infile = os.path.join(datasetPath, nameTrain_x)
    train_x = np.load(infile)
    infile = os.path.join(datasetPath, nameTrain_y)
    train_y = np.load(infile)
    infile = os.path.join(datasetPath, nameVal_x)
    val_x = np.load(infile)
    infile = os.path.join(datasetPath, nameVal_y)
    val_y = np.load(infile)
    infile = os.path.join(datasetPath, nameTest_x)
    test_x = np.load(infile)
    infile = os.path.join(datasetPath, nameTest_y)
    test_y = np.load(infile)
    infile = os.path.join(datasetPath, nameTest_id)
    test_id = np.load(infile)
    return train_x, train_y, val_x, val_y, test_x, test_y, test_id