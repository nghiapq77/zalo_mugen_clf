from subprocess import Popen, PIPE, STDOUT
from PIL import Image
from random import shuffle
from sklearn.utils import class_weight

import csv
import os
import gc
import eyed3
import numpy as np
import librosa as lbr

from config import genres
from configMode1 import trainCsvPath, audioPath, spectrogramPath, slicePath, datasetPath
from configMode1 import pixelPerSecond, sliceSize, testRatio, validationRatio
from configMode1 import nameTrain_x, nameTrain_y, nameVal_x, nameVal_y, nameTest_x, nameTest_y, nameTest_id

########################### Process Mp3  ###############################
########################################################################
currentPath = os.path.dirname(os.path.realpath(__file__))

def isMono(filename):
	audiofile = eyed3.load(filename)
	return audiofile.info.mode == 'Mono'

def createSingleSpectrogram(auPath, filename, newfilename, targetPath):
    if not os.path.exists(os.path.dirname(targetPath)):
		try:
			os.makedirs(os.path.dirname(targetPath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise    
    if not os.path.exists(os.path.join(targetPath, newfilename[:-4]+".png")): ###check if spectrogram exists alr
        if isMono(auPath+filename):
		    command = "cp '{}' '/tmp/{}'".format(auPath+filename,filename)
        else:
	        command = "sox '{}' '/tmp/{}' remix 1,2".format(auPath+filename,filename)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)

        output, errors = p.communicate()
        if errors:
	        print errors

        #Create spectrogram
        command = "sox '/tmp/{}' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(filename,pixelPerSecond,targetPath+newfilename[:-4])
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    
        output, errors = p.communicate()
        if errors:
	        print errors

        #Remove tmp mono track
        os.remove("/tmp/{}".format(filename))

def createMulSpectrogramTrain():
    with open(trainCsvPath, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            newfilename = rows[1]+"_"+rows[0]
            createSingleSpectrogram(audioPath, rows[0], newfilename, spectrogramPath)

####################### Process spectrogram ############################
########################################################################

def createSingleSlice(specPath, filename, targetPath):
    img = Image.open(specPath+filename)
    width, height = img.size
    nSamples = int(width/sliceSize)
    if not os.path.exists(os.path.dirname(targetPath)):
        try:
            os.makedirs(os.path.dirname(targetPath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    for i in range(nSamples):
        startPixel = i*sliceSize
        imgTmp = img.crop((startPixel, 1, startPixel+sliceSize, sliceSize+1))
        imgTmp.save(targetPath+"{}_{}.png".format(filename[:-4], i))

def createMulSlicesTrain():
    for filename in os.listdir(spectrogramPath):
        if filename.endswith(".png"):
            targetPath = slicePath+filename.split('_')[0] + '/'
            createSingleSlice(spectrogramPath, filename, targetPath)

########################### Process slice ##############################
########################################################################

def createDataset():
    data = []
    for genre in genres:
        #get slices
        filenames = os.listdir(slicePath+genre)
        filenames = [filename for filename in filenames]
        for filename in filenames:
            if not filename.endswith(".png"):
                continue
            img = Image.open(slicePath+genre+"/"+filename)
            imgData = np.asarray(img, dtype=np.uint8)
            imgData = imgData/255.
            label = [1. if genre == g else 0. for g in genres]
            data.append((imgData, label, filename.split('_')[1]))

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

    #Saving data
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
