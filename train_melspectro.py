import os
import csv
from melspectro import createSlicesFromAudio, getDataset
from model import createKerasModel
from config import nEpoch, batchSize, sliceSize, validationRatio
import librosa as lbr

#getting genres
csvfilepath = os.path.join("data/", "genres.csv")
genres = []
with open(csvfilepath, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        genres.append(int(rows[0]))
nClasses = len(genres)

slicePath = "data/train_full/slices/"
#creating slices
if not os.path.exists(slicePath):
    createSlicesFromAudio("data/train_full/", genres, sliceSize, validationRatio)
#getting features and labels
train_x, train_y, validation_x, validation_y = getDataset(slicePath, genres, sliceSize, validationRatio)
#normalizing features
print("Normalizing features with L2")
train_x = lbr.util.normalize(train_x, norm=2)
validation_x = lbr.util.normalize(validation_x, norm=2)

#creating model
model = createKerasModel(sliceSize, nClasses)
# Take a look at the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=nEpoch, validation_data=(validation_x, validation_y), batch_size=batchSize)
model.save('keras_model.h5')