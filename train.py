import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch, learningRate
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel, convertOutputToGenre, saveAsMat
import csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","createSlices","testDataInput","non","test","saveDataAsMat"])
args = parser.parse_args()

if "createSlices" in args.mode:
    createSpectrogramsFromAudio("data/train_full/", "train.csv")
    createSlicesFromSpectrograms("data/train_full/spectrograms/", sliceSize)

slicePath = "data/train_full/spectrograms/slices/"
genres = os.listdir(slicePath)
genres = [filename for filename in genres if os.path.isdir(slicePath+filename)]
nClasses = len(genres)

train_x, train_y, validation_x, validation_y = getDataset(slicePath, genres, sliceSize, validationRatio)

if "saveDataAsMat" in args.mode:
    saveAsMat(train_x,'train_x')
    saveAsMat(train_y,'train_y')
    saveAsMat(validation_x,'validation_x')
    saveAsMat(validation_y,'validation_y.]')

model = createModel(nClasses, sliceSize, learningRate)



if "train" in args.mode:
    #Train the model
    print("[+] Training the model...")
    run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))
    model.fit(train_x, train_y, n_epoch=nEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_x, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
    pred = model.predict(validation_x)
    print(convertOutputToGenre(pred))
    print("    Model trained!")
    #Save trained model
    print("[+] Saving the weights...")
    model.save('zalo_mugen_clf_model.tflearn')
    print("[+] Weights saved!")

if "test" in args.mode:
    model.load("zalo_mugen_clf_model.tflearn")
    pred = model.predict(train_x)


print("--- %s seconds ---" % (time.time() - start_time))