from tensorflow import keras
def createKerasModel(imageSize,nClasses):
    model =   keras.Sequential([
        keras.layers.Conv2D(
            input_shape=[imageSize,imageSize,1],
            filters=64,
            kernel_size=2,
            padding="same",
            activation="elu"
        ),
        keras.layers.MaxPooling2D(
            pool_size=2
        ),
        keras.layers.Conv2D(
            filters=128,
            kernel_size=2,
            padding="same",
            activation="elu"
        ),
        keras.layers.MaxPooling2D(
            pool_size=2
        ),
        keras.layers.Conv2D(
            filters=256,
            kernel_size=2,
            padding="same",
            activation="elu"
        ),
        keras.layers.MaxPooling2D(
            pool_size=2
        ),
        keras.layers.Conv2D(
            filters=512,
            kernel_size=2,
            padding="same",
            activation="elu"
        ),
        keras.layers.MaxPooling2D(
            pool_size=2
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(nClasses, activation='softmax')
    ])
    return model
