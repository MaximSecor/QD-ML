#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:07:20 2021

@author: maximsecor
"""

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import time

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
    
    grid = 32
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
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return ground_state_PES_1

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
   
grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
potential = generate_pot_test(grid_size,0,0,1500,1500,10)

n_states = 5
basis = generate_basis(xlist,potential,n_states)
prop = generate_propagator(xlist,potential,1000)

acting_potential = []
non_stationaty_states_present = []
non_stationaty_states_future = []

start = time.time()

for i in range(1000):

    potential = generate_pot(grid_size)
    prop = generate_propagator(xlist,potential,1000)
    basis = generate_basis(xlist,potential,n_states)
    state = generate_state(basis,n_states)

    for j in range(1):

        acting_potential.append(potential)
        state_RP = state
        non_stationaty_states_present.append(state_RP)
        state_future = np.matmul(prop,state_RP)
        non_stationaty_states_future.append(state_future)
        state = state_future

acting_potential = np.array(acting_potential)
non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)

end = time.time()
print("Datagen Time: ", end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)
present = np.concatenate((acting_potential,present),1)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

file_present = '/Users/maximsecor/Desktop/present.csv'
file_future = '/Users/maximsecor/Desktop/future.csv'

os.system('touch ' + file_present)
os.system('touch ' + file_future)

df_present = pd.DataFrame(present)
df_future = pd.DataFrame(future)

df_present.to_csv(file_present, index = False, header=True)
df_future.to_csv(file_future, index = False, header=True)

#%%

grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 5

test_potentials = []
test_states_real = []
test_states_imag = []

for i in range(1000):
    
    potential = generate_pot(grid_size)
    basis = generate_basis(xlist,potential,n_states)
    state = generate_state(basis,n_states)
    
    test_potentials.append(potential)
    test_states_real.append(np.real(state))
    test_states_imag.append(np.imag(state))

test_potentials = np.array(test_potentials)
test_states_real = np.array(test_states_real)
test_states_imag = np.array(test_states_imag)

#%%

file_test_potentials = '/Users/maximsecor/Desktop/file_test_potentials.csv'
os.system('touch ' + file_test_potentials)
df_file_test_potentials = pd.DataFrame(test_potentials)
df_file_test_potentials.to_csv(file_test_potentials, index = False, header=True)

file_test_states_real = '/Users/maximsecor/Desktop/file_test_states_real.csv'
os.system('touch ' + file_test_states_real)
df_file_test_states_real = pd.DataFrame(test_states_real)
df_file_test_states_real.to_csv(file_test_states_real, index = False, header=True)

file_test_states_imag = '/Users/maximsecor/Desktop/file_test_states_imag.csv'
os.system('touch ' + file_test_states_imag)
df_file_test_states_imag = pd.DataFrame(test_states_imag)
df_file_test_states_imag.to_csv(file_test_states_imag, index = False, header=True)

#%%

start = time.time()

grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 1
trials = 1000
trial_time = 1000

test_TD_potentials = []
for i in range(trials):
    potential = generate_pot_TD(trial_time)[0]
    test_TD_potentials.append(potential)
test_TD_potentials = np.array(test_TD_potentials)

end = time.time()
print(end-start)

#%%

print(test_TD_potentials.shape)
test_TD_potentials = (test_TD_potentials.reshape(trials,trial_time*grid_size))

#%%

file_test_TD_potentials = '/Users/maximsecor/Desktop/file_test_TD_potentials.csv'
os.system('touch ' + file_test_TD_potentials)
df_file_test_TD_potentials = pd.DataFrame(test_TD_potentials)
df_file_test_TD_potentials.to_csv(file_test_TD_potentials, index = False, header=True)
    
