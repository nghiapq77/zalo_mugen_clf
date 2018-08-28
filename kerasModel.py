from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
def createKerasModel(imageSize,nClasses):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=[128,128,1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(10,activation='softmax'))
    model.add(Activation('sigmoid'))

    return model
