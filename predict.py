import configMode2 as conf
from config import genres
from kerasModel import createKerasModel
from libMode2 import createSlicesFromAudio, createDataset, getDataset
import numpy as np
from sklearn.metrics import f1_score

nClasses = len(genres)
print("[+] Creating model")
model = createKerasModel(conf.sliceSize,nClasses)
model.load_weights('1_conv_32_nodes_0_dense_1535469438.h5')
model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
    )
print("[+] Getting dataset")
train_x, train_y, validation_x, validation_y, test_x, test_y , test_id = getDataset()
print("[+] Predicting")
pred = model.predict(test_x)
print(f1_score(test_y, pred, average=None))
print(f1_score(test_y, pred, average='macro'))
test_y = test_y.argmax(axis=1)
pred = pred.argmax(axis=1)
print(1*(test_y==pred))
print(np.mean(1*(test_y==pred)))