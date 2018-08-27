#Path
trainCsvPath = "data/train.csv"
datasetPath = "data/mode2/dataset/"
audioPath = "../data/train_full/"
slicePath = "data/mode2/slice/"
extractedFolderPath = "data/mode2/extractedFolder/"

#Name file
nameTrain_x = 'train_x.npy'
nameTrain_y = 'train_y.npy'

nameVal_x = 'val_x.npy'
nameVal_y = 'val_y.npy'

nameTest_x = 'test_x.npy'
nameTest_y = 'test_y.npy'
nameTest_id = 'test_id.npy'

#Spectrogram resolution
pixelPerSecond = 10

#Slice parameters
sliceSize = 128

#Genres
genres = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

#dataset param
validationRatio = 0.2
testRatio = 0.2

#Model parameters
batchSize = 128
learningRate = 0.000001
nEpoch = 5