import os
import csv
from melspectro import createSlicesFromAudio, getDataset
from model import createKerasModel
from config import nEpoch, batchSize, sliceSize, validationRatio

csvfilepath = os.path.join("data/", "genres.csv")
genres = []
with open(csvfilepath, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        genres.append(int(rows[0]))
nClasses = len(genres)
print(nClasses)
#createSlicesFromAudio("data/train_full/", genres, sliceSize, validationRatio)
train_x, train_y, validation_x, validation_y = getDataset("data/train/slices/", genres, sliceSize, validationRatio)

model = createKerasModel(128,nClasses)
#model.load_weights('keras_model.h5')

# Take a look at the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=nEpoch, validation_data=(validation_x, validation_y), batch_size=batchSize)
model.save('keras_model.h5')