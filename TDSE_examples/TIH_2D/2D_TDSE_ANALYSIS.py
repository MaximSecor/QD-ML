#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:01:31 2020

@author: maximsecor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
import os
import sys
import seaborn as sns; sns.set()

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

from scipy.signal import argrelextrema

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

np.set_printoptions(precision=4,suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import matplotlib.animation as animation
from celluloid import Camera

def get_den(state):
    return np.real(state*np.conj(state))

#%%

n_states = 5
mass = 1836
grid_size = 32

test_pot_1 = generate_pot()
basis = generate_basis(test_pot_1)
prop = generate_propagator(test_pot_2)
state_0 = generate_state(basis)

xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
ylist = np.linspace(-0.75,0.75,grid_size)*1.8897

X, Y = np.meshgrid(xlist,ylist)

timeseries_true = []
timeseries_true.append(get_den(state_0))

state = state_0
for i in range(250):
    
    state_future = np.matmul(prop,state)
    state = state_future
    state_den = get_den(state)
    timeseries_true.append(state_den)

timeseries_true = np.array(timeseries_true)
    
model = load_model('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/saved_model')

timeseries_pred = []
timeseries_pred.append(get_den(state_0))

state = state_0
state = state.reshape(1,-1)

state_real = np.real(state)
state_imag = np.imag(state)
state = np.concatenate((state_real,state_imag),1)

for i in range(250):
    state = model.predict(state)
    state_packed = state[0,:grid_size**2]+(state[0,grid_size**2:]*1j)
    state_den = get_den(state_packed)
    state = state/np.sqrt(np.sum(state_den))
    state_den = state_den * (1/np.sum(state_den))
    timeseries_pred.append(state_den)
    
timeseries_pred = np.array(timeseries_pred)

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,timeseries_true[i].reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('True Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_true.gif', writer = 'imagemagick')

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,timeseries_pred[i].reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Artificial Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_pred.gif', writer = 'imagemagick') 

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,(np.abs(timeseries_pred[i]-timeseries_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Trajectory Error')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_err.gif', writer = 'imagemagick') 
    
    
    
    
    
    
    
    
    

