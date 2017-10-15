# General Imports
import copy

# Keras imports
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, LSTM, Bidirectional
from keras.layers import Dense, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout
from keras import optimizers
from keras import backend as K

# Callbacks for training
from keras.callbacks import TensorBoard, EarlyStopping

# Import metrics functions
from sklearn.metrics import cohen_kappa_score

# Ploting
import matplotlib.pyplot as plt
from matplotlib.pyplot import stem
# %matplotlib inline


# Old imports #
import pandas as pd
import numpy as np

print("Start read")
data = pd.read_csv("TrumpTweets.csv")
print("Shape of dataset: "+str(data.shape))
print(data.head(25))
print('Hello world!')
