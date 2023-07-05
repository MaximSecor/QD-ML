#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:12:01 2020

@author: maximsecor
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressor

np.set_printoptions(precision=4,suppress=True)

import tensorflow as tf

from sklearn import preprocessing

import time
import os
import pandas as pd

import seaborn as sns; sns.set()

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

import matplotlib.animation as animation
from celluloid import Camera

#%%

# FGH function, its faster jitted. Remove jit if you dont want it,
@jit(nopython=True)
def fgh(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = np.zeros((nx,nx))
    tmat = np.zeros((nx,nx))
    hmat = np.zeros((nx,nx))
    
    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i,j] = potential[j]
                tmat[i,j] = (k**2)/3
            else:
                dji = j-i
                vmat[i,j] = 0
                tmat[i,j] = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
            hmat[i,j] = (1/(2*mass))*tmat[i,j] + vmat[i,j]
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%
    
# FGH function, its faster jitted. Remove jit if you dont want it,
@jit(nopython=True)
def prop_ham(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = np.zeros((nx,nx))
    tmat = np.zeros((nx,nx))
    hmat = np.zeros((nx,nx))
    
    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i,j] = potential[j]
                tmat[i,j] = (k**2)/3
            else:
                dji = j-i
                vmat[i,j] = 0
                tmat[i,j] = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
            hmat[i,j] = (1/(2*mass))*tmat[i,j] + vmat[i,j]

    return hmat

#%%
    
def get_den(state):
    return np.real(state*np.conj(state))

#%%
    
k = 1 
mass = 1
xlist = np.linspace(-10,10,64)
xpot = 0.5*k*xlist**2
pr_solutions = fgh(xlist,xpot,mass)

wvfns = pr_solutions[1].T
ener = pr_solutions[0]
print(ener[0:10])

#%%

n_states = 2
state_coeff = 2*np.random.random(n_states)-1
state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
state = np.matmul(state_coeff,wvfns[0:n_states])

hamiltonian = prop_ham(xlist,xpot,mass)
delta_t = 0.1
prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)
state_future = np.matmul(prop,state)

plot_den = []

for i in range(100):
    state_future = np.matmul(prop,state_future)
    plot_den.append(get_den(state_future))
    plt.ylim(-0.01,0.2)
    plt.plot(xlist,get_den(state_future))
    plt.show()
    
#%%
    
disc = 4
xlist_disc = np.linspace(-10,10,64) + disc
xpot_disc = 0.5*k*xlist_disc**2
pr_solutions = fgh(xlist_disc,xpot_disc,mass)
wvfns_disc = pr_solutions[1].T

state_coeff = np.array([1,1,1,1,1])
state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
state_disc = np.matmul(state_coeff,wvfns[0:len(state_coeff)])

plt.plot(state_disc)

#%%

disc_state = np.matmul(wvfns[0:n_states],wvfns_disc)
print(np.sum(disc_state**2))

#%%

non_stationaty_states_present = []
non_stationaty_states_future = []

for i in range(2000):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
    state = np.matmul(state_coeff,wvfns[0:n_states])

    for j in range(500):
        
        phase = (2*np.random.random()-1)*np.pi
        state_RP = state * np.exp(np.complex(0,1)*phase)
                
        non_stationaty_states_present.append(state_RP)
        state_future = np.matmul(prop,state_RP)
        
        non_stationaty_states_future.append(state_future)
        state = state_future
        
#%%

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)

print(present.shape)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

print(future.shape)

#%%

######################################
### Predict the Future
######################################

train = present
target = future

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = np.random.randint(100))

NN_model = Sequential()

NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)

val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
        
for i in range(13):
    NN_model.fit(train_tf, target_tf, epochs=8, batch_size=64*(i**2), verbose=2)


#%%
    
NN_model.save('/Users/maximsecor/Desktop/TDSE/saved_model_25')    

#%%
    
import time
start = time.time()

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_tf)
MAE = mean_absolute_error(target_tf, predictions_train)
print('Training Set Error =', MAE)

end = time.time()
print((end-start)/len(train))

#%%

grid_size = 64
future_pred = predictions_val[:,:grid_size]+(predictions_val[:,grid_size:]*np.complex(0,1))
future_true = val_y[:,:grid_size]+(val_y[:,grid_size:]*np.complex(0,1))

#%%

i = 1000
plt.plot(xlist,get_den(future_pred[i]),xlist,get_den(future_true[i]))
plt.show

#%%

plt.plot(xlist,future_pred[0]*np.conj(future_pred[0]))
plt.show
    
#%%

# n_states = 25
# state_coeff = 2*np.random.random(n_states)-1
# state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
# random_state = np.matmul(state_coeff,wvfns[0:n_states])
# random_state = wvfns_disc

# state_coeff = np.array([1,1,1,1,1])
# state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
# state_disc = np.matmul(state_coeff,wvfns_disc[0:len(state_coeff)])
# random_state = state_disc

# eigen = 4
# random_state = wvfns[eigen]

pos = 28
random_state = np.zeros((64))
random_state[pos] = 1

plot_den_true = []
plot_den_true.append(get_den(random_state))
teststep = random_state

for i in range(250):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,get_den(teststep))
    # plt.show
    
plot_den_pred = []
plot_den_pred.append(get_den(random_state))
teststep = random_state
teststep = teststep.reshape(1,-1)

teststep_real = np.real(teststep)
teststep_imag = np.imag(teststep)
teststep = np.concatenate((teststep_real,teststep_imag),1)

for i in range(250):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,test_den)
    # plt.show
    
    
# for i in range(250):
#     plt.ylim(-0.01,0.4)
#     plt.plot(xlist,xpot*0.01,xlist,plot_den_true[i],xlist,plot_den_pred[i])
#     plt.show()

    
fig = plt.figure(dpi=600)
camera = Camera(fig)
for i in range(250):
    plt.ylim(-0.01,0.4)
    plt.plot(xlist, xpot*0.01, color = 'black')
    plt.plot(xlist, plot_den_true[i], color = 'red')
    plt.plot(xlist, plot_den_pred[i], color = 'blue')
    camera.snap()
animation = camera.animate(interval=50)
animation.save('/Users/maximsecor/Desktop/TDSE_movie_pos28_true25.gif', writer = 'imagemagick')   
    
    
    
    
    