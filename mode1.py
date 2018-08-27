import configMode1 as conf
from config import genres
from kerasModel import createKerasModel
from libMode1 import createMulSpectrogramTrain, createMulSlicesTrain, createDataset, getDataset
from lib import getClassWeight

def printConfig():
    print("=========================================================================")
    print("\t\t\t *** CONFIG ***")
    print("[+] Path of data")
    print("\tAudio path: {}".format(conf.audioPath))
    print("\tDataset path: {}".format(conf.datasetPath))
    print("\tSpectrogram path: {}".format(conf.spectrogramPath))
    print("\tSlice path: {}".format(conf.slicePath))

    print("[+] Model neural network ")
    print("\tValidation ratio: {}".format(conf.validationRatio))
    print("\tSlice size: {}".format(conf.sliceSize))
    print("\tLearning rate: {}".format(conf.learningRate))
    print("\tNumber of epoch : {}".format(conf.nEpoch))
    print("=========================================================================")

def createSpectrogram():
    print('[+] Creating spectrogram')
    createMulSpectrogramTrain()

def createSlice():
    print('[+] Creating slice')
    createMulSlicesTrain()

def createData():
    print('[+] Creating data')
    createDataset()

def train():
    nClasses = len(genres)
    print("[+] Creating model")
    model = createKerasModel(conf.sliceSize,nClasses)
    model.load_weights('keras_model.h5')
    model.summary()
    print("[+] Getting dataset")
    train_x, train_y, validation_x, validation_y, test_x, test_y , test_id = getDataset()
    class_weights = getClassWeight(train_y)
    print("[+] Fitting model")

    model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

    model.fit(
        train_x, train_y,
        epochs=conf.nEpoch,
        validation_data=(validation_x, validation_y),
        batch_size=conf.batchSize,
        class_weight=class_weights
    )
    model.save('keras_model.h5')
