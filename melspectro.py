""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
""""""
import time
start_time = time.time()
import os
import csv
import numpy as np
import librosa as lbr
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}
import errno
from config import validationRatio, sliceSize
from subprocess import Popen, PIPE, STDOUT

#Define current path
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

def createSlicesFromAudio(filename, size):
    tmp, _ = createFeaturesFromAudio(filename)
    nSamples = int(tmp.shape[0]/size)
    slices = []
    for i in range(nSamples):
        start = i*size
        slice_i = tmp[start:start+size][:size]
        slices.append(slice_i)
    return slices

def createDatasetFromAudio(audioPath, genres, sliceSize, validationRatio):
    slicePath = os.path.join(audioPath, "slices/")
    if not os.path.exists(slicePath):
        os.mkdir(slicePath)
    extractedFolder = os.path.join(audioPath, "extractedMP3/")
    if not os.path.exists(extractedFolder):
        os.mkdir(extractedFolder)
    csvfilename = "train.csv"
    csvfilepath = os.path.join(audioPath, csvfilename)
    with open(csvfilepath, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            genre = int(rows[1])
            name = rows[0]
            audiofilepath = os.path.join(audioPath, name)
            if os.path.exists(audiofilepath):
                slices = createSlicesFromAudio(audiofilepath, sliceSize)
                label = [1. if genre == g else 0. for g in genres]
                i = 0
                for slice_i in slices:
                    data = (slice_i, label)
                    outfile = os.path.join(slicePath, "{}_{}.npy".format(name[:-4], i))
                    np.save(outfile, data)
                    i+=1
                #move trained mp3 to another folder
                command = "mv {} {}".format(audiofilepath, extractedFolder)
                p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
                _, errors = p.communicate()
                if errors:
                    print errors
                print("Finished processing file {}".format(name))

def getDataset(audioPath, genres, sliceSize, validationRatio):
    if not os.path.isfile(audioPath+"train_x.npy"):
        print("Creating dataset")
        createDatasetFromAudio(audioPath, genres, sliceSize, validationRatio)
    else:
        print("Using existing dataset")
    #TODO
    #get all train/validating data
    
    return train_x, train_y, validation_x, validation_y

csvfilepath = os.path.join("data/train/", "genres.csv")
genres = []
with open(csvfilepath, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        genres.append(int(rows[0]))
createDatasetFromAudio("data/train/", genres, sliceSize, validationRatio)

print("--- %s seconds ---" % (time.time() - start_time))