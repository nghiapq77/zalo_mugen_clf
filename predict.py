import configPredict as conf
from config import genres
from kerasModel import createKerasModel
from libPredict import createSlicesForPredict, createDatasetForPredict, getDatasetForPredict
from lib import getClassWeight

def createSlice():
    print('[+] Creating slice')
    createSlicesForPredict()

def createData():
    print('[+] Creating data')
    createDatasetForPredict()