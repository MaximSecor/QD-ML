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

# test_pot_2 = generate_pot()
# prop = generate_propagator(test_pot_2)
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
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,test_pot_2*627,xlist,state_den*100)
    # plt.show()

timeseries_true = np.array(timeseries_true)
  
model = load_model('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TIH_1D/saved_model')

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
    
    tstep = i
    
    plt.rcParams["figure.figsize"] = (6,4)
    
    plt.text(1.1, 45, str(str(int(tstep))+" fs"), color="k", fontsize=18)
    
    plt.ylim(-1,50)
    plt.plot(xlist,test_pot_2[0]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b')
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('True Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TIH_1D/1D_HBOND_true.gif', writer = 'imagemagick')

#%%

n_states = 5

test_pot_1 = generate_pot()
basis = generate_basis(test_pot_1)
basis_2 = generate_basis(test_pot_2.reshape(32))

print(np.matmul(basis,basis.T))
state_0 = generate_state(basis)
print(np.matmul(state_0,basis_2.T))

state_pred = state_0
state_true = state_0

state_pred = state_pred.reshape(1,-1)

state_pred_real = np.real(state_pred)
state_pred_real = np.imag(state_pred)
state_pred = np.concatenate((state_pred_real,state_pred_real),1)

#%%

for i in range(5):

    state_true = np.matmul(prop,state_true)
    coeff_pred = np.matmul(state_true,basis_2.T)
    # print(coeff_pred*np.conj(coeff_pred))
    # print(np.sum(np.real(coeff_pred*np.conj(coeff_pred))*energies))
    
    # print(state*)

#%%

total_time = 100
trajectories = 25
n_states = 5
report_times = np.array([1,5,10,50,100])-1

overlap_error = np.zeros((total_time))
energy_error = np.zeros((total_time))
coeffs_error = np.zeros((total_time,n_states))

for j in range(trajectories):
    
    print(j)

    test_pot_1 = generate_pot()
    basis = generate_basis(test_pot_1)
    basis_2 = generate_basis(test_pot_2.reshape(32))
    
    # print(np.matmul(basis,basis.T))
    state_0 = generate_state(basis)
    # print(np.sum(np.matmul(state_0,basis_2.T)**2))
    
    state_pred = 0
    state_pred = state_0
    state_true = state_0
    
    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    
    coeff_true = np.matmul(state_true,basis_2.T)
    energies = fgh_1D(xlist,test_pot_2.reshape(32),mass)[0][:n_states]*627
    energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
    # print(energy_true)
    
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
        coeff_true = np.matmul(state_true,basis_2.T)
        energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
        
        # print(get_den(np.sum(state_pred_packed*np.conj(state_true))))
        
        overlap_error[i] = overlap_error[i] + (1-np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true)))))
        energy_error[i] = energy_error[i] + (energy_pred-energy_true)
        coeffs_error[i] = coeffs_error[i] + (coeff_pred-coeff_true)

print('\n', overlap_error[report_times]/trajectories)
print('\n', energy_error[report_times]/trajectories)
print('\n', coeffs_error[report_times]/trajectories)

#%%

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

#%%

total_time = 1
trajectories = 1
n_states_init = 1
n_states_anal = 10

report_times = np.array([1])-1

overlap_error = np.zeros((total_time))
energy_error = np.zeros((total_time))
coeffs_error = np.zeros((total_time,n_states_anal))

#%%

for j in range(trajectories):
    
    print(j)

    test_pot_1 = generate_pot()
    
    basis_1 = generate_basis(test_pot_1,n_states_init)
    basis_2 = generate_basis(test_pot_2.reshape(32),n_states_anal)
    
    # print(np.matmul(basis_1,basis_1.T))
    state_0 = generate_state(basis_1,n_states_init)
    # print(np.sum(np.matmul(state_0,basis_2.T)**2))
    
    state_pred = 0
    state_pred = state_0
    state_true = state_0
    
    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    
    coeff_true = np.matmul(state_true,basis_2.T)
    energies = fgh_1D(xlist,test_pot_2.reshape(32),mass)[0][:n_states_anal]*627
    energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
    # print(energy_true)
    
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
        
        print(prop.shape)
        print(state_true.shape)
        print(basis_2.shape)
        
        state_true = np.matmul(state_true,basis_2.T)
        energy_true = (np.sum(np.real(coeff_true*np.conj(coeff_true))*energies))
        
        # # print(get_den(np.sum(state_pred_packed*np.conj(state_true))))
        
        overlap_error[i] = overlap_error[i] + (1-np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true)))))
        # energy_error[i] = energy_error[i] + (energy_pred-energy_true)
        # coeffs_error[i] = coeffs_error[i] + (coeff_pred-coeff_true)
    
    # print(state_true)

print('\n', overlap_error[report_times]/trajectories)
print('\n', energy_error[report_times]/trajectories)
print('\n', coeffs_error[report_times]/trajectories)

#%%
    
energies = fgh_1D(xlist,test_pot_2.reshape(32),mass)[0][:5]*627
print(energies)
    
#%%

    


    

