# let's compare simple model with 1 simpleRNN vs. deep model with 2 simpleRNN
# using timeSeries data: simply create with numpy! (sine function)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def load_data(num_data, window_size):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, num_data, 1)
    
    time = np.linspace(0, 1, window_size + 1)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.1 * np.sin((time - offsets2) * (freq2 * 10 + 10))
    series += 0.1 * (np.random.rand(num_data, window_size + 1) - 0.5)
    
    num_train = int(num_data * 0.8)
    X_train, y_train = series[:num_train, :window_size], series[:num_train, -1]
    X_test, y_test = series[num_train:, :window_size], series[num_train:, -1]
    
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    
    return X_train, X_test, y_train, y_test

