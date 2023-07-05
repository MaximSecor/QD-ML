#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:25:22 2020

@author: maximsecor
"""

#%%

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

from keras.callbacks import ModelCheckpoint,EarlyStopping
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

def generate_basis(ground_state_PES_1):
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    temp = fgh_1D(xlist,ground_state_PES_1,mass)
    basis = temp[1][:,0:n_states].T
    return basis

def generate_state(basis):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 1000/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []
acting_hamiltonian = []

n_states = 5
mass = 1836
grid_size = 64

for k in range(1000):
    
    test_pot_1 = generate_pot()
    basis = generate_basis(test_pot_1)
    
    test_pot_2 = generate_pot()
    prop = generate_propagator(test_pot_2)
    
    for i in range(10):
        
        state = generate_state(basis)

        # rand_time = np.random.random()*(10**6)
        # hamiltonian = prop_1D(xlist,test_pot_2,mass)
        # delta_t = rand_time/24.18
        # rand_prop = expm(-1j*hamiltonian*delta_t)
        # state = np.matmul(rand_prop,state)

        for j in range(10):
            
            acting_hamiltonian.append(test_pot_2)
        
            phase = (2*np.random.random()-1)*np.pi
            state_RP = state * np.exp(1j*phase)
            
            non_stationaty_states_present.append(state_RP)
            state_future = np.matmul(prop,state_RP)
            non_stationaty_states_future.append(state_future)
            state = state_future
            
non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)
acting_hamiltonian = np.array(acting_hamiltonian)

end = time.time()
print(end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)
present = np.concatenate((acting_hamiltonian,present),1)

# print(present.shape)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

# print(future.shape)

train = present
target = future

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 1103)
train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

dense_layers = 2
dense_nodes = 768

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1)
model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.001))
model.compile(loss='mean_squared_error', optimizer=opt)

#%%

print("\nModel Configured")
start = time.time()
for i in range(9):
    model.fit(train_tf, target_tf, epochs=160000, batch_size=1280*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

#%%

# model.save('saved_model')
# model = load_model('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_1D/saved_model')

overlap_tracker = []
den_error_tracker = []

for q in range(10):
    start = time.time()
    
    sim_length = 100
    n_states = 1
    mass = 1836
    grid_size = 64
    
    dynamic_pot_list = []
    
    start_pot = generate_pot()
    test_pot_1 = start_pot
    
    for q in range(int(sim_length/25)):
    
        test_pot_2 = generate_pot()
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,test_pot_1*627,xlist,test_pot_2*627)
        # plt.show()
        
        for i in range(25):
            step = i*np.pi/50
            dynamic_pot = test_pot_1*np.cos(step)**2+test_pot_2*np.sin(step)**2
            dynamic_pot = dynamic_pot - np.min(dynamic_pot)
            dynamic_pot_list.append(dynamic_pot)
            # plt.ylim(-1,50)
            # plt.plot(xlist,dynamic_pot*627)
            # plt.show()
            
        test_pot_1 = test_pot_2
    
    dynamic_pot_list = np.array(dynamic_pot_list)
    
    # test_pot_1 = generate_pot()
    basis = generate_basis(start_pot)
    
    # test_pot_2 = generate_pot()
    # prop = generate_propagator(test_pot_2)
    
    state_0 = generate_state(basis)
    
    time_series_den_true = []
    time_series_state_true = []
    
    time_series_den_true.append(get_den(state_0))
    time_series_state_true.append(state_0)
    
    state = state_0
    for i in range(sim_length):
        
        prop = generate_propagator(dynamic_pot_list[i])
        state_future = np.matmul(prop,state)
        state = state_future
        state_den = get_den(state)
        
        time_series_den_true.append(state_den)
        time_series_state_true.append(state)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,dynamic_pot_list[i]*627,xlist,state_den*100)
        # plt.show()
    
    time_series_den_true = np.array(time_series_den_true)
    time_series_state_true = np.array(time_series_state_true)
    
    time_series_den_pred = []
    time_series_state_pred = []
    
    time_series_den_pred.append(get_den(state_0))
    time_series_state_pred.append(state_0)
    
    state = state_0
    state = state.reshape(1,-1)
    
    test_pot_2 = test_pot_2.reshape(1,-1)
    
    state_real = np.real(state)
    state_imag = np.imag(state)
    state = np.concatenate((state_real,state_imag),1)
    
    for i in range(sim_length):
        
        pot_temp = dynamic_pot_list[i]
        pot_temp = pot_temp.reshape(1,-1)
        state = np.concatenate((pot_temp,state),1)
        state = model.predict(state)
        state_packed = state[0,:grid_size]+(state[0,grid_size:]*1j)
        state_den = get_den(state_packed)
        
        state_packed = state_packed/np.sqrt(np.sum(state_den))
        state = state/np.sqrt(np.sum(state_den))
        state_den = state_den * (1/np.sum(state_den))
        
        time_series_den_pred.append(state_den)
        time_series_state_pred.append(state_packed)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,pot_temp[0]*627,xlist,timeseries_pred[i]*100)
        # plt.show()
        
    time_series_den_pred = np.array(time_series_den_pred)
    time_series_state_pred = np.array(time_series_state_pred)
    
    end = time.time()
    # print('\nTraining Time: ',(end-start))
    
    overlap_squared_list = []
    den_error = []
    
    for i in range(11):
        
        time_temp = sim_length/10
        
        overlap = np.sum(time_series_state_pred[int(time_temp*i)]*(np.conj(time_series_state_true[int(time_temp*i)])))
        overlap_squared = np.real(overlap*np.conj(overlap))
        overlap_squared_list.append(overlap_squared)
        
        den_error.append(np.sum(np.abs(time_series_den_pred[int(time_temp*i)] - time_series_den_true[int(time_temp*i)])))
    
    overlap_squared_list = np.array(overlap_squared_list)
    # print(overlap_squared_list)
    overlap_tracker.append(overlap_squared_list)
    
    den_error = np.array(den_error)
    # print(den_error)
    den_error_tracker.append(den_error)
    
    # overlap_squared_plot = []
    
    # for i in range(sim_length):
    #     overlap = np.sum(time_series_state_pred[i]*(np.conj(time_series_state_true[i])))
    #     overlap_squared = np.real(overlap*np.conj(overlap))
    #     overlap_squared_plot.append(overlap_squared)
        
    # overlap_squared_plot = np.array(overlap_squared_plot)
    
    # plt.plot(overlap_squared_plot)
    # plt.show()

overlap_tracker = np.array(overlap_tracker)
den_error_tracker = np.array(den_error_tracker)

print("\nContinuously changing potential")
print(np.mean(overlap_tracker,0))
print(np.mean(den_error_tracker,0))

overlap_tracker = []
den_error_tracker = []

for q in range(10):
    start = time.time()
    
    sim_length = 100
    n_states = 1
    mass = 1836
    grid_size = 64
    
    dynamic_pot_list = []
    
    start_pot = generate_pot()
    test_pot_1 = start_pot
    
    test_pot_2 = generate_pot()
    
    for q in range(int(sim_length/25)):
    
        # test_pot_2 = generate_pot()
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,test_pot_1*627,xlist,test_pot_2*627)
        # plt.show()
        
        for i in range(25):
            step = i*np.pi/50
            # dynamic_pot = test_pot_1*np.cos(step)**2+test_pot_2*np.sin(step)**2
            # dynamic_pot = dynamic_pot - np.min(dynamic_pot)
            dynamic_pot = test_pot_2
            dynamic_pot_list.append(dynamic_pot)
            # plt.ylim(-1,50)
            # plt.plot(xlist,dynamic_pot*627)
            # plt.show()
            
        # test_pot_1 = test_pot_2
    
    dynamic_pot_list = np.array(dynamic_pot_list)
    
    # test_pot_1 = generate_pot()
    basis = generate_basis(start_pot)
    
    # test_pot_2 = generate_pot()
    prop = generate_propagator(test_pot_2)
    
    state_0 = generate_state(basis)
    
    time_series_den_true = []
    time_series_state_true = []
    
    time_series_den_true.append(get_den(state_0))
    time_series_state_true.append(state_0)
    
    state = state_0
    for i in range(sim_length):
        
        # prop = generate_propagator(dynamic_pot_list[i])
        state_future = np.matmul(prop,state)
        state = state_future
        state_den = get_den(state)
        
        time_series_den_true.append(state_den)
        time_series_state_true.append(state)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,dynamic_pot_list[i]*627,xlist,state_den*100)
        # plt.show()
    
    time_series_den_true = np.array(time_series_den_true)
    time_series_state_true = np.array(time_series_state_true)
    
    time_series_den_pred = []
    time_series_state_pred = []
    
    time_series_den_pred.append(get_den(state_0))
    time_series_state_pred.append(state_0)
    
    state = state_0
    state = state.reshape(1,-1)
    
    test_pot_2 = test_pot_2.reshape(1,-1)
    
    state_real = np.real(state)
    state_imag = np.imag(state)
    state = np.concatenate((state_real,state_imag),1)
    
    for i in range(sim_length):
        
        pot_temp = dynamic_pot_list[i]
        pot_temp = pot_temp.reshape(1,-1)
        state = np.concatenate((pot_temp,state),1)
        state = model.predict(state)
        state_packed = state[0,:grid_size]+(state[0,grid_size:]*1j)
        state_den = get_den(state_packed)
        
        state_packed = state_packed/np.sqrt(np.sum(state_den))
        state = state/np.sqrt(np.sum(state_den))
        state_den = state_den * (1/np.sum(state_den))
        
        time_series_den_pred.append(state_den)
        time_series_state_pred.append(state_packed)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,pot_temp[0]*627,xlist,timeseries_pred[i]*100)
        # plt.show()
        
    time_series_den_pred = np.array(time_series_den_pred)
    time_series_state_pred = np.array(time_series_state_pred)
    
    end = time.time()
    # print('\nTraining Time: ',(end-start))
    
    overlap_squared_list = []
    den_error = []
    
    for i in range(11):
        
        time_temp = sim_length/10
        
        overlap = np.sum(time_series_state_pred[int(time_temp*i)]*(np.conj(time_series_state_true[int(time_temp*i)])))
        overlap_squared = np.real(overlap*np.conj(overlap))
        overlap_squared_list.append(overlap_squared)
        
        den_error.append(np.sum(np.abs(time_series_den_pred[int(time_temp*i)] - time_series_den_true[int(time_temp*i)])))
    
    overlap_squared_list = np.array(overlap_squared_list)
    # print(overlap_squared_list)
    overlap_tracker.append(overlap_squared_list)
    
    den_error = np.array(den_error)
    # print(den_error)
    den_error_tracker.append(den_error)
    
    # overlap_squared_plot = []
    
    # for i in range(sim_length):
    #     overlap = np.sum(time_series_state_pred[i]*(np.conj(time_series_state_true[i])))
    #     overlap_squared = np.real(overlap*np.conj(overlap))
    #     overlap_squared_plot.append(overlap_squared)
        
    # overlap_squared_plot = np.array(overlap_squared_plot)
    
    # plt.plot(overlap_squared_plot)
    # plt.show()

overlap_tracker = np.array(overlap_tracker)
den_error_tracker = np.array(den_error_tracker)

print("\nSuddenly changing potential")
print(np.mean(overlap_tracker,0))
print(np.mean(den_error_tracker,0))

overlap_tracker = []
den_error_tracker = []

for q in range(10):
    start = time.time()
    
    sim_length = 100
    n_states = 5
    mass = 1836
    grid_size = 64
    
    dynamic_pot_list = []
    
    start_pot = generate_pot()
    test_pot_1 = start_pot
    
    test_pot_2 = test_pot_1
    
    for q in range(int(sim_length/25)):
    
        # test_pot_2 = generate_pot()
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,test_pot_1*627,xlist,test_pot_2*627)
        # plt.show()
        
        for i in range(25):
            step = i*np.pi/50
            # dynamic_pot = test_pot_1*np.cos(step)**2+test_pot_2*np.sin(step)**2
            # dynamic_pot = dynamic_pot - np.min(dynamic_pot)
            dynamic_pot = test_pot_2
            dynamic_pot_list.append(dynamic_pot)
            # plt.ylim(-1,50)
            # plt.plot(xlist,dynamic_pot*627)
            # plt.show()
            
        # test_pot_1 = test_pot_2
    
    dynamic_pot_list = np.array(dynamic_pot_list)
    
    # test_pot_1 = generate_pot()
    basis = generate_basis(start_pot)
    
    # test_pot_2 = generate_pot()
    prop = generate_propagator(test_pot_2)
    
    state_0 = generate_state(basis)
    
    time_series_den_true = []
    time_series_state_true = []
    
    time_series_den_true.append(get_den(state_0))
    time_series_state_true.append(state_0)
    
    state = state_0
    for i in range(sim_length):
        
        # prop = generate_propagator(dynamic_pot_list[i])
        state_future = np.matmul(prop,state)
        state = state_future
        state_den = get_den(state)
        
        time_series_den_true.append(state_den)
        time_series_state_true.append(state)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,dynamic_pot_list[i]*627,xlist,state_den*100)
        # plt.show()
    
    time_series_den_true = np.array(time_series_den_true)
    time_series_state_true = np.array(time_series_state_true)
    
    time_series_den_pred = []
    time_series_state_pred = []
    
    time_series_den_pred.append(get_den(state_0))
    time_series_state_pred.append(state_0)
    
    state = state_0
    state = state.reshape(1,-1)
    
    test_pot_2 = test_pot_2.reshape(1,-1)
    
    state_real = np.real(state)
    state_imag = np.imag(state)
    state = np.concatenate((state_real,state_imag),1)
    
    for i in range(sim_length):
        
        pot_temp = dynamic_pot_list[i]
        pot_temp = pot_temp.reshape(1,-1)
        state = np.concatenate((pot_temp,state),1)
        state = model.predict(state)
        state_packed = state[0,:grid_size]+(state[0,grid_size:]*1j)
        state_den = get_den(state_packed)
        
        state_packed = state_packed/np.sqrt(np.sum(state_den))
        state = state/np.sqrt(np.sum(state_den))
        state_den = state_den * (1/np.sum(state_den))
        
        time_series_den_pred.append(state_den)
        time_series_state_pred.append(state_packed)
        
        # plt.ylim(-1,50)
        # plt.plot(xlist,pot_temp[0]*627,xlist,timeseries_pred[i]*100)
        # plt.show()
        
    time_series_den_pred = np.array(time_series_den_pred)
    time_series_state_pred = np.array(time_series_state_pred)
    
    end = time.time()
    # print('\nTraining Time: ',(end-start))
    
    overlap_squared_list = []
    den_error = []
    
    for i in range(11):
        
        time_temp = sim_length/10
        
        overlap = np.sum(time_series_state_pred[int(time_temp*i)]*(np.conj(time_series_state_true[int(time_temp*i)])))
        overlap_squared = np.real(overlap*np.conj(overlap))
        overlap_squared_list.append(overlap_squared)
        
        den_error.append(np.sum(np.abs(time_series_den_pred[int(time_temp*i)] - time_series_den_true[int(time_temp*i)])))
    
    overlap_squared_list = np.array(overlap_squared_list)
    # print(overlap_squared_list)
    overlap_tracker.append(overlap_squared_list)
    
    den_error = np.array(den_error)
    # print(den_error)
    den_error_tracker.append(den_error)
    
    # overlap_squared_plot = []
    
    # for i in range(sim_length):
    #     overlap = np.sum(time_series_state_pred[i]*(np.conj(time_series_state_true[i])))
    #     overlap_squared = np.real(overlap*np.conj(overlap))
    #     overlap_squared_plot.append(overlap_squared)
        
    # overlap_squared_plot = np.array(overlap_squared_plot)
    
    # plt.plot(overlap_squared_plot)
    # plt.show()

overlap_tracker = np.array(overlap_tracker)
den_error_tracker = np.array(den_error_tracker)

print("\nUnchanging potential")
print(np.mean(overlap_tracker,0))
print(np.mean(den_error_tracker,0))



