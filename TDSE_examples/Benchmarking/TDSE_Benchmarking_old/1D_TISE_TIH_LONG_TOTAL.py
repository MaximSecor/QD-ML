#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:13:48 2021

@author: maximsecor
"""

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam

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

import numpy as np
from numba import jit
import time
import os
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=4,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

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
from keras.callbacks import EarlyStopping

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

#%%

@jit(nopython=True)
def fgh_1D(domain,potential,mass):
    
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
    
@jit(nopython=True)
def prop_1D(domain,potential,mass):
    
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
    
def get_den(state):
    return np.real(state*np.conj(state))

def generate_pot():
    
    X0 = (0.5*np.random.random())
    KX_1 = (2500*np.random.random()+1500)/(350*627)
    KX_2 = (2500*np.random.random()+1500)/(350*627)
    dE = (20*np.random.random()-10)/627

    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    
    potential_1 = 0.5*mass*(KX_1**2)*(xlist-X0)**2
    potential_2 = 0.5*mass*(KX_2**2)*(xlist+X0)**2 + dE
    couplings = np.full((grid_size),10)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return ground_state_PES_1

def generate_basis(ground_state_PES_1,n_states):
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    temp = fgh_1D(xlist,ground_state_PES_1,mass)
    basis = temp[1][:,0:n_states].T
    return basis

def generate_state(basis,n_states):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 1*(1000/24.18)
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%
    
start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []
acting_hamiltonian = []

n_states = 5
mass = 1836
grid_size = 32
xlist = np.linspace(-0.75,0.75,grid_size)*1.8897

X0 = 0
KX_1 = (1500)/(350*627)
KX_2 = (1500)/(350*627)
dE = 0

potential_1 = 0.5*mass*(KX_1**2)*(xlist-X0)**2
potential_2 = 0.5*mass*(KX_2**2)*(xlist+X0)**2 + dE
couplings = np.full((grid_size),10)*(1/627)

two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
test_pot_2 = ground_state_PES - np.min(ground_state_PES)
prop = generate_propagator(test_pot_2)
  
for i in range(10000):
    
    test_pot_1 = generate_pot()
    basis = generate_basis(test_pot_1,n_states)
    state = generate_state(basis,n_states)

    for j in range(100):
    
        phase = (2*np.random.random()-1)*np.pi
        state_RP = state * np.exp(1j*phase)
        
        non_stationaty_states_present.append(state_RP)
        state_future = np.matmul(prop,state_RP)
        non_stationaty_states_future.append(state_future)
        state = state_future
        
non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)

end = time.time()
print(end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)

print(present.shape)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

print(future.shape)

#%%

features = present[:100000]
target = future[:100000]

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size = 0.1, random_state = 217)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1)

dense_layers = 2
dense_nodes = 512

model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.001))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

#%%

print("\nModel Configured")
start = time.time()
for i in range(13):
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))
    
#%%

# for i in range(13):
#     i=i
#     print(16*(2**i))
#     model.fit(train_X_tf, train_y_tf, epochs=64, batch_size=16*(2**i), verbose=2)

#%%
    
n_states = 5
mass = 1836
grid_size = 32

test_pot_1 = generate_pot()
basis = generate_basis(test_pot_1,n_states)

# test_pot_2 = generate_pot()
# prop = generate_propagator(test_pot_2)
state_0 = generate_state(basis,n_states)

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
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,test_pot_2*627,xlist,state_den*100)
    # plt.show()

timeseries_true = np.array(timeseries_true)

timeseries_pred = []
timeseries_pred.append(get_den(state_0))

state = state_0
state = state.reshape(1,-1)

test_pot_2 = test_pot_2.reshape(1,-1)

state_real = np.real(state)
state_imag = np.imag(state)
state = np.concatenate((state_real,state_imag),1)
# state = np.concatenate((test_pot_2,state),1)

for i in range(250):
    state = model.predict(state)
    state_packed = state[0,:grid_size]+(state[0,grid_size:]*1j)
    state_den = get_den(state_packed)
    state = state/np.sqrt(np.sum(state_den))
    state_den = state_den * (1/np.sum(state_den))
    timeseries_pred.append(state_den)
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,test_pot_2[0]*627,xlist,timeseries_pred[i]*100)
    # plt.show()
    
    # state = np.concatenate((test_pot_2,state),1)
    
timeseries_pred = np.array(timeseries_pred)

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i*50
    
    plt.rcParams["figure.figsize"] = (6,4)
    
    plt.text(1.1, 45, str(str(int(tstep))+" fs"), color="k", fontsize=18)
    
    plt.ylim(-1,50)
    plt.plot(xlist,test_pot_2[0]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b')
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('True Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=1000)
animation.save('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TIH_1D/1D_HBOND_VERY_LONG_true.gif', writer = 'imagemagick')

#%%

total_time = 50
trajectories = 10
n_states_init = 5
n_states_anal = 5

report_times = np.array([1,5,10,25,50])-1

overlap_error = np.zeros((total_time))
energy_error = np.zeros((total_time))
coeffs_error = np.zeros((total_time,n_states_anal))

for j in range(trajectories):
    
    print(j)

    test_pot_1 = generate_pot()
    
    basis_1 = generate_basis(test_pot_1,n_states_init)
    basis_2 = generate_basis(test_pot_2.reshape(32),n_states_anal)
    state_0 = generate_state(basis_1,n_states_init)
    # print(np.matmul(state_0,basis_2.T)**2)
    
    state_pred = state_0
    state_true = state_0
    
    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    
    coeff_true = np.matmul(state_true,basis_2.T)
    energies = fgh_1D(xlist,test_pot_2.reshape(32),mass)[0][:n_states_anal]*627
    energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
    
    for i in range(total_time):
        
        state_pred = model.predict(state_pred)
        state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
        state_pred_den = get_den(state_pred_packed)
        state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
        state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
        state_pred_den = state_pred_den * (1/np.sum(state_pred_den))
        
        # print(np.sum(state_pred_packed*np.conj(state_pred_packed)))
        coeff_pred = np.matmul(state_pred_packed,basis_2.T)
        # print(coeff_pred*np.conj(coeff_pred))
        energy_pred = (np.sum(np.real(coeff_pred*np.conj(coeff_pred))*energies))
        
        state_true = np.matmul(prop,state_true)
        
        # print(prop.shape)
        # print(state_true.shape)
        # print(basis_2.shape)
        
        coeff_true = np.matmul(state_true,basis_2.T)
        energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
        
        # # print(get_den(np.sum(state_pred_packed*np.conj(state_true))))
        
        # print(np.sum(state_pred_packed*np.conj(state_true)))
        
        overlap_error[i] = overlap_error[i] + (1-np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true)))))
        energy_error[i] = energy_error[i] + (energy_pred-energy_true)
        coeffs_error[i] = coeffs_error[i] + (get_den(coeff_pred)-get_den(coeff_true))
    
    # print(state_true)

print('\n', overlap_error[report_times]/trajectories)
print('\n', energy_error[report_times]/trajectories)
print('\n', coeffs_error[report_times]/trajectories)

#%%

truip = np.array([1,2,3,4,5,6,7])
print(truip[:3])
