#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:48:04 2020

@author: maximsecor
"""

#%%

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=4,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

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

def generate_pot(length):
    
    grid = 32
    mass = 1836
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    KX_1 = (2500*np.random.random()+1500)/(350*627)
    KX_2 = (2500*np.random.random()+1500)/(350*627)
    dE = (10*np.random.random()-5)/627
    
    red_mass = ((12*12)/(12+12))*1836
    coord_eq = 0.25*np.random.random()
    X0 = 2*coord_eq*np.random.random()
    x_da = X0
    v_da = 0
    a_da = 0
    freq_da = 200*np.random.random()+100
    k_const = red_mass*(freq_da/(627*350))**2
    dt_da = 1/0.024188
    
    f0 = 55.734
    mu = 0.2652
    lam = (5*np.random.random())
    x_solv = np.sqrt(lam/(2*f0))
    freq_solv = np.sqrt(f0/mu)
    dt_solv = 0.001

    potential_trajectory = []
    
    for t in range(length):
        
        x_da = x_da + v_da*dt_da + 0.5*a_da*dt_da**2
        a_da_2 = a_da
        a_da = -k_const*(x_da-coord_eq)/red_mass
        v_da = v_da + 0.5*(a_da_2+a_da)*dt_da
        X0 = x_da
         
        x_solv_t = x_solv*np.cos(freq_solv*t*dt_solv)
        solv_gap = x_solv_t*(np.sqrt(2*f0*lam))
        
        potential_1 = 0.5*mass*(KX_1**2)*(xlist-(X0))**2
        potential_2 = 0.5*mass*(KX_2**2)*(xlist+(X0))**2 + dE + solv_gap*(1/627)
        couplings = np.full((grid_size),10)*(1/627)
        
        two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
        ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
        ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
        
        potential_trajectory.append(ground_state_PES_1)
        
    potential_trajectory = np.array(potential_trajectory)
    
    return potential_trajectory

def generate_basis(potential,n_states):
    
    mass = 1836
    grid_size = 32
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    
    temp = fgh_1D(xlist,potential,mass)
    basis = temp[1][:,0:n_states].T
    
    return basis

def generate_state(basis,n_states):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator(potential):
    
    mass = 1836
    grid_size = 32
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    
    hamiltonian = prop_1D(xlist,potential,mass)
    delta_t = 1000/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

data_tests = np.array([1,10,100,1000,10000])
data_tests_inv = data_tests[::-1]
report_times = np.array([1,3,10,30,100,300,1000])-1

full_trials = 1
n_states = 5
mass = 1836
grid_size = 32

complete_overlap_error = np.zeros((len(data_tests),len(report_times)))

start_total = time.time()

for q_big in range(full_trials):
    
    start_sub = time.time()
    print("\nPotential Number: ", q_big)
    
    specific_potential = generate_pot(1000)[np.random.randint(1000)]
    prop = generate_propagator(specific_potential)

    for k in range(len(data_tests)):
        
        non_stationaty_states_present = []
        non_stationaty_states_future = []
        
        start = time.time()
    
        for i in range(data_tests[k]):
            
            packet_potential = generate_pot(1000)
            basis = generate_basis(packet_potential[np.random.randint(1000)],n_states)
            state = generate_state(basis,n_states)
    
            for j in range(data_tests_inv[k]):
            
                phase = (2*np.random.random()-1)*np.pi
                state_RP = state * np.exp(1j*phase)
                
                non_stationaty_states_present.append(state_RP)
                state_future = np.matmul(prop,state_RP)
                non_stationaty_states_future.append(state_future)
                state = state_future
    
        non_stationaty_states_present = np.array(non_stationaty_states_present)
        non_stationaty_states_future = np.array(non_stationaty_states_future)
        
        end = time.time()
        # print("Datagen Time: ", end-start)
        
        present_real = np.real(non_stationaty_states_present)
        present_imag = np.imag(non_stationaty_states_present)
        present = np.concatenate((present_real,present_imag),1)
        future_real = np.real(non_stationaty_states_future)
        future_imag = np.imag(non_stationaty_states_future)
        future = np.concatenate((future_real,future_imag),1)
        
        features = present
        target = future
        
        dense_layers = 2
        dense_nodes = 512
        learning_rates = 0.0001
        
        train_X, val_X, train_y, val_y = train_test_split(features, target, test_size = 0.1, random_state = 217)
        
        train_X_tf = tf.convert_to_tensor(train_X, np.float32)
        train_y_tf = tf.convert_to_tensor(train_y, np.float32)
        val_X_tf = tf.convert_to_tensor(val_X, np.float32)
        val_y_tf = tf.convert_to_tensor(val_y, np.float32)
        
        overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 10)
        
        model = Sequential()
        model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
        for layers in range(dense_layers):
            model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
        model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
        opt = Adam(learning_rate=(learning_rates))
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
        
        start = time.time()
        for i in range(13):
            model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=0, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
        end = time.time()
        # print('Training Time: ',(end-start))
        
        total_time = 1000
        trajectories = 1
        n_states_init = 5
        n_states_anal = 10
        
        overlap_error = np.zeros((total_time))
        prop = generate_propagator(specific_potential)
        
        start = time.time()
        
        for j in range(trajectories):
        
            packet_potential = generate_pot(1000)
            basis_1 = generate_basis(packet_potential[np.random.randint(1000)],n_states_init)    
            state_0 = generate_state(basis_1,n_states_init)
            state_pred = state_0
            state_true = state_0
        
            state_pred = state_pred.reshape(1,-1)
            state_pred_real = np.real(state_pred)
            state_pred_imag = np.imag(state_pred)
            state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
            state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
        
            for i in range(total_time):
        
                state_pred = model.predict(state_pred)
                state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
                state_pred_den = get_den(state_pred_packed)
                state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
                state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
                state_pred_den = state_pred_den * (1/np.sum(state_pred_den))

                state_true = np.matmul(prop,state_true)
        
                overlap_error[i] = overlap_error[i] + (np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true)))))
                
        end = time.time()
        # print('Evaluation Time: ' ,(end-start))
                    
        end_sub = time.time()
        print(overlap_error[report_times]/(trajectories))
        
        complete_overlap_error[k,:] = complete_overlap_error[k,:] + overlap_error[report_times]/(trajectories)
    
    end_total = time.time()
    print('Sub Time: ',(end_sub-start_sub))

print("\nDone!")
print(complete_overlap_error/full_trials)
print("Time Total: ", (end_total-start_total))

#%%

for i in range(100):
    specific_potential = generate_pot(1000)[np.random.randint(1000)]
    plt.ylim(-1,50)
    plt.plot(specific_potential*627)
    plt.show()
    
#%%
    
specific_potential = generate_pot(1000)

for i in range(10):
    plt.ylim(-1,50)
    plt.plot(specific_potential[i*10]*627)
plt.show()
    
    