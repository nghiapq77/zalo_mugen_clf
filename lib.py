from sklearn.utils import class_weight
import numpy as np

def getClassWeight(train_y):
    class_weights = train_y.argmax(axis=1)
    class_weights = np.concatenate((np.asarray(range(10)),class_weights),axis=None)
    class_weights = class_weight.compute_class_weight('balanced',np.unique(class_weights),class_weights)
    class_weights = dict(enumerate(class_weights))
    return class_weights