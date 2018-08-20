import os
import random
import string
import time
start_time = time.time()
from config import sliceSize, validationRatio, batchSize, nEpoch, learningRate
from songToData import createSlicesFromSpectrograms, createSpectrogramsFromAudio
from model import getDataset, createModel, convertOutputToGenre, saveAsMat
import csv

#createSpectrogramsFromAudio("data/train_full/", "train.csv")
#createSlicesFromSpectrograms("data/train_full/spectrograms/", sliceSize)

slicePath = "data/train_full/spectrograms/slices/"
genres = os.listdir(slicePath)
genres = [filename for filename in genres if os.path.isdir(slicePath+filename)]
nClasses = len(genres)
print(genres)

train_x, train_y, validation_x, validation_y = getDataset(slicePath, genres, sliceSize, validationRatio)

model = createModel(nClasses, sliceSize)

#Train the model
print("[+] Training the model...")
run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))
model.fit(train_x, train_y, n_epoch=nEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_x, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
print("    Model trained!")
#Save trained model
print("[+] Saving the weights...")
model.save('zalo_mugen_clf_model.tflearn')
print("[+] Weights saved!")

print("--- %s seconds ---" % (time.time() - start_time))
