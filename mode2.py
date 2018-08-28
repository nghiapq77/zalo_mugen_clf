import configMode2 as conf
from config import genres
from kerasModel import createKerasModel
from libMode2 import createSlicesFromAudio, createDataset, getDataset
from lib import getClassWeight

def printConfig():
    print("=========================================================================")
    print("\t\t\t *** CONFIG ***")
    print("[+] Path of data")
    print("\tAudio path: {}".format(conf.audioPath))
    print("\tDataset path: {}".format(conf.datasetPath))

    print("[+] Model neural network ")
    print("\tValidation ratio: {}".format(conf.validationRatio))
    print("\tSlice size: {}".format(conf.sliceSize))
    print("\tLearning rate: {}".format(conf.learningRate))
    print("\tNumber of epoch : {}".format(conf.nEpoch))
    print("=========================================================================")


def createSlice():
    print('[+] Creating slice')
    createSlicesFromAudio()

def createData():
    print('[+] Creating data')
    createDataset()

def train():
    nClasses = len(genres)
    print("[+] Creating model")
    model = createKerasModel(conf.sliceSize,nClasses)
    model.load_weights('1_conv_32_nodes_0_dense_1535469438.h5')
    model.summary()
    print("[+] Getting dataset")
    train_x, train_y, validation_x, validation_y, test_x, test_y , test_id = getDataset()
    print("[+] Fitting model")

    model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

    model.fit(
        train_x, train_y,
        epochs=conf.nEpoch,
        validation_data=(validation_x, validation_y),
        batch_size=conf.batchSize,
    )
    model.save('data/mode2/keras_model.h5')