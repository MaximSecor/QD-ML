#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:03:47 2021

@author: maximsecor
"""

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import time
import cmath
import sys

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=10,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import tensorflow as tf

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

def generate_pot_TD(length):

    dt_fs = 1000/24.188
    
    grid_size = 32
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    mass = 1836
    K_p1 = mass*((2500*np.random.random()+1500)/(350*627))**2
    K_p2 = mass*((2500*np.random.random()+1500)/(350*627))**2
    
    R_freq = ((200*np.random.random()+100)/(627*350))
    R_eq = (0.8*np.random.random()+0.2)*1.8897
    R_0 = (2.0*R_eq*np.random.random())
    
    f0 = 55.734
    mu = 0.2652
    lam = (10*np.random.random())
    X_freq = np.sqrt(f0/mu)*0.001
    X_eq = (20*np.random.random()-10)
    
    phase_1 = 2*np.pi*np.random.random()
    phase_2 = 2*np.pi*np.random.random()

    potential_trajectory = []
    test_pos = []
    test_solv = []
    
    for t in range(length):
        
        R_coord = (R_0-R_eq)*(np.cos(R_freq*t*dt_fs+phase_1))+R_eq
        X_coord = (lam)*(np.cos(X_freq*t+phase_2))+X_eq
        
        potential_1 = 0.5*K_p1*(xlist-(0.5*R_coord))**2
        potential_2 = 0.5*K_p2*(xlist+(0.5*R_coord))**2 + X_coord*(1/627)
        couplings = np.full((grid_size),10)*(1/627)
        
        two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
        ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
        ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
        
        potential_trajectory.append(ground_state_PES_1)
        test_pos.append(R_coord)
        test_solv.append(X_coord*(1/627))
        
    potential_trajectory = np.array(potential_trajectory)
    test_pos = np.array(test_pos)
    test_solv = np.array(test_solv)
    
    potential_1 = 0.5*K_p1*(xlist-(0.5*R_eq))**2
    potential_2 = 0.5*K_p2*(xlist+(0.5*R_eq))**2 + X_eq*(1/627)
    couplings = np.full((grid_size),10)*(1/627)
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return potential_trajectory,test_pos,test_solv,ground_state_PES_1,R_eq,R_0

def generate_pot(grid_size):
    
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    mass = 1836
    K_p1 = mass*((2500*np.random.random()+1500)/(350*627))**2
    K_p2 = mass*((2500*np.random.random()+1500)/(350*627))**2
    
    R_eq = (2*np.random.random())*1.8897
    X_eq = (40*np.random.random()-20)
    
    # R_eq = (0.5*np.random.random())*1.8897
    # X_eq = (10*np.random.random()-20)
    
    potential_1 = 0.5*K_p1*(xlist-(0.5*R_eq))**2
    potential_2 = 0.5*K_p2*(xlist+(0.5*R_eq))**2 + X_eq*(1/627)
    couplings = np.full((grid_size),10)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return ground_state_PES_1

def generate_pot_test(grid_size,X_eq,R_eq,w_p1,w_p2,couplings):
    
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    mass = 1836
    K_p1 = mass*((w_p1)/(350*627))**2
    K_p2 = mass*((w_p2)/(350*627))**2
    
    potential_1 = 0.5*K_p1*(xlist-(0.5*R_eq))**2
    potential_2 = 0.5*K_p2*(xlist+(0.5*R_eq))**2 + X_eq*(1/627)
    couplings = np.full((grid_size),couplings)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    potential_gs = np.linalg.eigh(two_state.T)[0].T[0]
    potential_es = np.linalg.eigh(two_state.T)[0].T[1]
    
    return potential_gs,potential_es

def generate_basis(xlist,potential,n_states):
    
    temp = fgh_1D(xlist,potential,1836)
    basis = temp[1][:,0:n_states].T
    
    return basis

def generate_state(basis,n_states):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    
    state_phase = (2*np.random.random(n_states))*np.pi
    state_coeff = state_coeff * np.exp(1j*state_phase)
    
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator(xlist,potential,dt):
    
    hamiltonian = prop_1D(xlist,potential,1836)
    delta_t = dt/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%
    
print("\nLoading Data")

file_present = '/Users/maximsecor/Desktop/present.csv'
file_future = '/Users/maximsecor/Desktop/future.csv'

data_present = pd.read_csv(file_present)
data_future = pd.read_csv(file_future)

present = data_present.values
future = data_future.values

features = present
target = future

print("Loading Data Complete")

#%%

dense_layers = 0
dense_nodes = 128
learning_rates = 0.001

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size = 0.1, random_state = 217)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1)

model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(learning_rates))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

start = time.time()
for i in range(3):
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('Training Time: ',(end-start))

#%%

print("\nComplex Error")

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y_tf, predictions_val)
print('Cross-Validation Set Error =', MAE)

print("\nDensity Error")

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(get_den(train_y_tf), get_den(predictions_train))
print('Training Set Error =', MAE)

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(get_den(val_y_tf), get_den(predictions_val))
print('Cross-Validation Set Error =', MAE)

print("\nOverlap Error")

state_pred_packed = predictions_train[:,:grid_size]+(predictions_train[:,grid_size:]*1j)
state_true_packed = train_y[:,:grid_size]+(train_y[:,grid_size:]*1j)
mae_overlap = []
for i in range(state_pred_packed.shape[0]):
    state_pred_packed[i] = state_pred_packed[i] / np.sqrt(np.sum(state_pred_packed[i]*np.conj(state_pred_packed[i])))
    overlap = (np.sum(state_pred_packed[i]*np.conj(state_true_packed[i])))
    num = cmath.polar(overlap)
    mae_overlap.append(abs(np.array(num)))
mae_overlap = np.array(mae_overlap)    
print('Training Set Error =', np.mean(mae_overlap,0))

state_pred_packed = predictions_val[:,:grid_size]+(predictions_val[:,grid_size:]*1j)
state_true_packed = val_y[:,:grid_size]+(val_y[:,grid_size:]*1j)
mae_overlap = []
for i in range(state_pred_packed.shape[0]):
    state_pred_packed[i] = state_pred_packed[i] / np.sqrt(np.sum(state_pred_packed[i]*np.conj(state_pred_packed[i])))
    overlap = (np.sum(state_pred_packed[i]*np.conj(state_true_packed[i])))
    num = cmath.polar(overlap)
    mae_overlap.append(abs(np.array(num)))
mae_overlap = np.array(mae_overlap)    
print('Cross-Validation Set Error =', np.mean(mae_overlap,0))    

#%%

file_test_potentials = '/Users/maximsecor/Desktop/file_test_potentials.csv'
data_test_potentials = pd.read_csv(file_test_potentials)
test_states_potentials = data_test_potentials.values

file_test_states_real = '/Users/maximsecor/Desktop/file_test_states_real.csv'
data_test_states_real = pd.read_csv(file_test_states_real)
test_states_real = data_test_states_real.values

file_test_states_imag = '/Users/maximsecor/Desktop/file_test_states_imag.csv'
data_test_states_imag = pd.read_csv(file_test_states_imag)
test_states_imag = data_test_states_imag.values

test_states = test_states_real + test_states_imag*1j

#%%

start = time.time()

total_time = 100
trials = 1
overlap_error_total = np.zeros((total_time,2))
n_states = 5

for q in range(trials):
    
    print(q)

    potential = test_states_potentials[q]
    state = test_states[q]
    prop = generate_propagator(xlist,potential,1000)
    
    state_pred = state
    state_true = state
    
    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
    
    overlap_error = np.zeros((total_time,2))
    
    for i in range(total_time):
    
        ann_in = np.concatenate((potential.reshape(-1,1).T,state_pred),1)
        state_pred = model.predict(ann_in)
        state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
        state_pred_den = get_den(state_pred_packed)
        state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
        state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
        state_pred_den = state_pred_den * (1/np.sum(state_pred_den))
    
        state_true = np.matmul(prop,state_true)
        overlap_error[i] = overlap_error[i] + np.array(np.abs(cmath.polar(np.sum(state_pred_packed*np.conj(state_true)))))

    overlap_error_total += overlap_error
    
end = time.time()
print(end-start)

for i in range(total_time):
    print("Time: ",i,"Error: ",overlap_error_total[i]/trials)

print("\nTime:",1," Acc:",overlap_error_total[0]/trials)
print("Time:",10," Acc:",overlap_error_total[9]/trials)
print("Time:",100," Acc:",overlap_error_total[99]/trials)
# print("Time:",250," Acc:",rs_overlap_error_total[249]/trials)
# print("Time:",500," Acc:",rs_overlap_error_total[499]/trials)
# print("Time:",1000," Acc:",rs_overlap_error_total[999]/trials)

#%%

file_test_TD_potentials = '/Users/maximsecor/Desktop/file_test_TD_potentials.csv'
data_test_TD_potentials = pd.read_csv(file_test_TD_potentials)
test_TD_potentials = data_test_TD_potentials.values
test_TD_potentials = (test_TD_potentials.reshape(1000,1000,32))

#%%

start = time.time()

total_time = 100
trials = 1
overlap_error_total = np.zeros((total_time,2))
n_states = 1

for q in range(trials):

    print(q)
    
    potential = test_TD_potentials[q]
    basis = generate_basis(xlist,potential[0],n_states)
    state = generate_state(basis,n_states)
    
    state_pred = state
    state_true = state

    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
    
    overlap_error = np.zeros((total_time,2))

    for i in range(total_time):
    
        ann_in = np.concatenate((potential[i].reshape(-1,1).T,state_pred),1)
        state_pred = model.predict(ann_in)
        state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
        state_pred_den = get_den(state_pred_packed)
        state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
        state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
        state_pred_den = state_pred_den * (1/np.sum(state_pred_den))
    
        prop = generate_propagator(xlist,potential[i],1000)
        state_true = np.matmul(prop,state_true)
        overlap_error[i] = overlap_error[i] + np.array(np.abs(cmath.polar(np.sum(state_pred_packed*np.conj(state_true)))))

    overlap_error_total += overlap_error
    
end = time.time()
print(end-start)

for i in range(total_time): 
    print("Time: ",i,"Error: ",overlap_error_total[i]/trials)
    
print("\nTime:",1," Acc:",overlap_error_total[0]/trials)
print("Time:",10," Acc:",overlap_error_total[9]/trials)
print("Time:",100," Acc:",overlap_error_total[99]/trials)
# print("Time:",250," Acc:",rs_overlap_error_total[249]/trials)
# print("Time:",500," Acc:",rs_overlap_error_total[499]/trials)
# print("Time:",1000," Acc:",rs_overlap_error_total[999]/trials)


    
    