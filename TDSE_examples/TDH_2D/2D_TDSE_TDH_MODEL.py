#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:38:00 2020

@author: maximsecor
"""

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressor

np.set_printoptions(precision=4,suppress=True)

import tensorflow as tf

import time
import os
import pandas as pd

#%%

file_present = '/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_2D/file_present.csv'
file_future = '/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_2D/file_future.csv'

data_present = pd.read_csv(file_present)
data_future = pd.read_csv(file_future)

present = data_present.values
future = data_future.values

#%%

train = present[:10]
target = future[:10]

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 1103)
train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

dense_layers = 4
dense_nodes = 2048

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1)
model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.0001))
model.compile(loss='mean_squared_error', optimizer=opt)

#%%

print("\nModel Configured")
start = time.time()
for i in range(9):
    model.fit(train_tf, target_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

# model.save('saved_model')

#%%

for i in range(8):
    print(16*(2**i))
    model.fit(train_tf, target_tf, epochs=32, batch_size=16*(2**i), verbose=2)

#%%
    
model.fit(train_tf, target_tf, epochs=4096, batch_size=9, verbose=2)

#%%

model.save('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_2D/saved_model')

















