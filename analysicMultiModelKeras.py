from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from random import shuffle

import time
import numpy as np

from libMode2 import getDataset


train_x, train_y, validation_x, validation_y, test_x, test_y , test_id = getDataset()
print("[+] Load data successfully")

denseLayers = [0, 1, 2]
layerSizes = [32, 64, 128]
convLayers = [1, 2, 3]

idRandom = np.arange(train_x.shape[0])
shuffle(idRandom)
idRandom = idRandom[10000:]

for denseLayer in denseLayers:
    for layerSize in layerSizes:
        for convLayer in convLayers:
            NAME = "{}_conv_{}_nodes_{}_dense_{}".format(convLayer, layerSize, denseLayer, int(time.time()))
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layerSize, (3, 3), input_shape=train_x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(convLayer-1):
                model.add(Conv2D(layerSize, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(denseLayer):
                model.add(Dense(layerSize))
                model.add(Activation('relu'))

            model.add(Dense(10,activation='softmax'))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(
                            loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'],
                        )
            model.fit(
                        train_x, train_y,
                        validation_data=(validation_x, validation_y),
                        batch_size=128,
                        epochs=3,
                        validation_split=0.2,
                        callbacks=[tensorboard]
                    )
            model.save(NAME+'.h5')