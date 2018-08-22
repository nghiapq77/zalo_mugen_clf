""" warning ignore """
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
""""""
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
from config import validationRatio, sliceSize, melspectroDatasetPath
from subprocess import Popen, PIPE, STDOUT
from random import shuffle

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

def sliceAudio(filename, size):
    tmp, _ = createFeaturesFromAudio(filename)
    nSamples = int(tmp.shape[0]/size)
    slices = []
    for i in range(nSamples):
        start = i*size
        slice_i = tmp[start:start+size][:size]
        slices.append(slice_i)
    return slices

def createSlicesFromAudio(audioPath, genres, sliceSize, validationRatio):
    #creating folders
    slicePath = os.path.join(audioPath, "slices/")
    if not os.path.exists(slicePath):
        os.mkdir(slicePath)
    for g in genres:
        path = os.path.join(slicePath, "{}".format(g))
        if not os.path.exists(path):
            os.mkdir(path)
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
                slices = sliceAudio(audiofilepath, sliceSize)
                label = [1. if genre == g else 0. for g in genres]
                i = 0
                for slice_i in slices:
                    data = (slice_i, label)
                    outfile = os.path.join(slicePath, "{}/{}_{}.npy".format(genre, name[:-4], i))
                    np.save(outfile, data)
                    i+=1
                #move trained mp3 to another folder
                command = "mv {} {}".format(audiofilepath, extractedFolder)
                p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
                _, errors = p.communicate()
                if errors:
                    print errors
                print("Finished processing file {}".format(name))

def createDatasetFromSlices(slicePath, genres, sliceSize, validationRatio):
    if not os.path.exists(melspectroDatasetPath):
        os.mkdir(melspectroDatasetPath)
    data = []
    for genre in genres:
        filenames = os.listdir(slicePath+str(genre))
        filenames = [filename for filename in filenames]
        #capping maximum number of slices
        cappedSlices = len(filenames)/20
        shuffle(filenames)
        filenames = filenames[:cappedSlices]
        for i in range(cappedSlices):
            infile = os.path.join(slicePath, "{}/{}".format(genre, filenames[i]))
            data.append(np.load(infile))
    shuffle(data)
    x, y = zip(*data)
    nValidation = int(len(x)*validationRatio)
    nTrain = len(x)-nValidation
    train_x = np.asarray(x[:nTrain]).reshape([-1, sliceSize, sliceSize, 1])
    train_y = np.asarray(y[:nTrain])
    validation_x = np.asarray(x[-nValidation:]).reshape([-1, sliceSize, sliceSize, 1])
    validation_y = np.asarray(y[-nValidation:])
    #saving
    outfile = os.path.join(melspectroDatasetPath, "train_x.npy")
    np.save(outfile, train_x)
    outfile = os.path.join(melspectroDatasetPath, "train_y.npy")
    np.save(outfile, train_y)
    outfile = os.path.join(melspectroDatasetPath, "validation_x.npy")
    np.save(outfile, validation_x)
    outfile = os.path.join(melspectroDatasetPath, "validation_y.npy")
    np.save(outfile, validation_y)

def getDataset(slicePath, genres, sliceSize, validationRatio):
    if not os.path.isfile(melspectroDatasetPath+"train_x.npy"):
        print("[+] Creating dataset with slices of size {}".format(sliceSize))
        createDatasetFromSlices(slicePath, genres, sliceSize, validationRatio) 
    else:
        print("[+] Using existing dataset")
    #loading dataset
    infile = os.path.join(melspectroDatasetPath, "train_x.npy")
    train_x = np.load(infile)
    infile = os.path.join(melspectroDatasetPath, "train_y.npy")
    train_y = np.load(infile)
    infile = os.path.join(melspectroDatasetPath, "validation_x.npy")
    validation_x = np.load(infile)
    infile = os.path.join(melspectroDatasetPath, "validation_y.npy")
    validation_y = np.load(infile)
    return train_x, train_y, validation_x, validation_y
"""
csvfilepath = os.path.join("data/", "genres.csv")
genres = []
with open(csvfilepath, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        genres.append(int(rows[0]))
createSlicesFromAudio("data/train_full/", genres, sliceSize, validationRatio)
"""