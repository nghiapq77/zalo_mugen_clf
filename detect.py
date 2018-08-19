import os
import random
import string
import time
start_time = time.time()
import numpy as np
from config import sliceSize, validationRatio, batchSize, nEpoch, learningRate
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel, convertOutputToGenre, saveAsMat
import csv

slicePath = "data/train_full/spectrograms/slices/"
genres = os.listdir(slicePath)
genres = [filename for filename in genres if os.path.isdir(slicePath+filename)]
nClasses = len(genres)
train_x, train_y, validation_x, validation_y = getDataset(slicePath, genres, sliceSize, validationRatio)

model = createModel(nClasses, sliceSize, learningRate)
listNumTrainExample = np.arange(50)*100
listErr =  np.array([len(a),4])
for i in 1:50:
    
    model = createModel(nClasses, sliceSize, learningRate)
    print("[+] Training the model...")
    run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))
    model.fit(train_x, train_y, n_epoch=nEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_x, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
    pred = model.predict(validation_x)
    print(convertOutputToGenre(pred))
    print("    Model trained!")

