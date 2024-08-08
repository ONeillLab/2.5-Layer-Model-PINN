import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from keras import Model
from name_list import *
import keras
from netCDF4 import Dataset
import time
from random import sample


model = keras.models.load_model("NPINNmain.keras")
for k in range(200):
    for i in range(256):
        u1_inp = np.random.random((1,N,N))
        v1_inp = np.random.random((1,N,N))
        u2_inp = np.random.random((1,N,N))
        v2_inp = np.random.random((1,N,N))
        h1_inp = np.random.random((1,N,N))
        h2_inp = np.random.random((1,N,N))
        Wmat_inp = np.random.random((1,N,N))
        uvh = model([u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp]) 
    print(k)