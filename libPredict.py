from random import shuffle

import gc
import os
import csv
import numpy as np 
import librosa as lbr

from config import genres
from configPredict import testCsvPath, audioPath, slicePath, datasetPath
from configPredict import sliceSize
from configPredict import namePredict_x

WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

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

def createSlicesForPredict():
    #creating folders
    if not os.path.exists(slicePath):
        os.mkdir(slicePath)
    with open(testCsvPath, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            name = rows[0]
            audioFilePath = os.path.join(audioPath, name)
            if os.path.exists(audioFilePath):
                feature, _ = createFeaturesFromAudio(audioFilePath)
                nSamples = int(feature.shape[0]/sliceSize)
                for i in range(nSamples):
                    start = i*sliceSize
                    slice = feature[start:start+sliceSize][:sliceSize]
                    outfile = os.path.join(slicePath, "{}_{}.npy".format(name[:-4], i))
                    np.save(outfile, slice)
            print("Finished processing file {}".format(name))

def createDatasetForPredict():
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)
    predict_x = []
    filenames = os.listdir(slicePath)
    filenames = [filename for filename in filenames]
    for filename in filenames:
        if not filename.endswith(".npy"):
            continue
        infile = os.path.join(slicePath, "{}".format(filename))
        npData = np.load(infile)
        npData = lbr.util.normalize(npData, norm=2)
        predict_x.append(npData)

    #saving
    print("[+] Saving dataset")
    outfile = os.path.join(datasetPath, namePredict_x)
    np.save(outfile, predict_x)
    return predict_x

########################### Process dataset ############################
########################################################################

def checkExistDataset():
    dataExist = True
    if not os.path.isfile(datasetPath+namePredict_x):
        print("\t predict_x data is not exist")
        dataExist = False
    return dataExist

def getDatasetForPredict():
    if not checkExistDataset() :
        print("[+] Creating new dataset")
        return createDatasetForPredict() 
    else:
        print("[+] Loading exist dataset")
    infile = os.path.join(datasetPath, namePredict_x)
    predict_x = np.load(infile)
    return predict_x