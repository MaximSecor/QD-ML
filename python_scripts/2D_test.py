#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:49:11 2021

@author: maximsecor
"""

#%%

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import sys

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=4,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
import seaborn as sns; sns.set()

#%%

@jit(nopython=True)
def fgh_2d(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    hmat = np.zeros((nx**2,nx**2))
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if (xi == xj) & (yi == yj):
                        vmat = potential[xj,yj]
                        tmat = (k**2)*(2/3)
                    if (xi == xj) & (yi != yj):
                        dji = yj - yi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi == yj):
                        dji = xj - xi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi != yj):
                        vmat = 0
                        tmat = 0
                    hmat[xi+yi*nx,xj+yj*nx] = (1/(2*mass))*tmat+vmat
                        
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

@jit(nopython=True)
def pot_2d(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    hmat = np.zeros((nx**2,nx**2))
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if (xi == xj) & (yi == yj):
                        vmat = potential[xj,yj]
                        tmat = (k**2)*(2/3)
                    if (xi == xj) & (yi != yj):
                        dji = yj - yi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi == yj):
                        dji = xj - xi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi != yj):
                        vmat = 0
                        tmat = 0
                    hmat[xi+yi*nx,xj+yj*nx] = (1/(2*mass))*tmat+vmat
                        
    return hmat
    
def get_den(state):
    return np.real(state*np.conj(state))

def generate_pot():
    
    X0 = (0.5*np.random.random())
    
    KX_1 = (2500*np.random.random()+1500)/(350*627)
    KX_2 = (2500*np.random.random()+1500)/(350*627)
    KY_1 = (2500*np.random.random()+1500)/(350*627)
    KY_2 = (2500*np.random.random()+1500)/(350*627)
    dE = (20*np.random.random()-10)/627

    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    ylist = np.linspace(-0.75,0.75,grid_size)*1.8897
    
    X, Y = np.meshgrid(xlist,ylist)
    
    potential_1 = 0.5*mass*(KX_1**2)*(X-X0)**2 + 0.5*mass*(KY_1**2)*Y**2 
    potential_2 = 0.5*mass*(KX_2**2)*(X+X0)**2 + 0.5*mass*(KY_2**2)*Y**2 + dE
    couplings = np.full((grid_size,grid_size),10)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return ground_state_PES_1

def generate_pot_2(KX_1,KX_2,KY_1,KY_2,X0,dE):
    
    # X0 = (0.5*np.random.random())
    # KX_1 = (2500*np.random.random()+1500)/(350*627)
    # KX_2 = (2500*np.random.random()+1500)/(350*627)
    # KY_1 = (2500*np.random.random()+1500)/(350*627)
    # KY_2 = (2500*np.random.random()+1500)/(350*627)
    # dE = (20*np.random.random()-10)/627

    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    ylist = np.linspace(-0.75,0.75,grid_size)*1.8897
    
    X, Y = np.meshgrid(xlist,ylist)
    
    potential_1 = 0.5*mass*(KX_1**2)*(X-X0)**2 + 0.5*mass*(KY_1**2)*Y**2 
    potential_2 = 0.5*mass*(KX_2**2)*(X+X0)**2 + 0.5*mass*(KY_2**2)*Y**2 + dE
    couplings = np.full((grid_size,grid_size),10)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return ground_state_PES_1

def generate_basis(ground_state_PES_1):
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    temp = fgh_2d(xlist,ground_state_PES_1,mass)
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

def generate_propagator(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = pot_2d(xlist,test_pot_2,mass)
    delta_t = 1000/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []
acting_hamiltonian = []

n_states = 5
mass = 1836
grid_size = 8

test_pot_2 = generate_pot()
prop = generate_propagator(test_pot_2)
basis = generate_basis(test_pot_2)

for k in range(100000):
    
    for i in range(1):
        
        state = generate_state(basis,n_states)

        for j in range(1):
            
            non_stationaty_states_present.append(state)
            state = np.matmul(prop,state)
            non_stationaty_states_future.append(state)
            
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

features = present
target = future

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size = 0.1, random_state = np.random.randint(100))

model = Sequential()
model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    model.add(Dense(512, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.001))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

for i in range(8):
    print(16*(2**i))
    model.fit(train_tf, target_tf, epochs=12, batch_size=16*(2**i), verbose=2)

#%%

state_0 = generate_state(basis,n_states)

state_true = state_0
state_pred = state_0

timeseries_true = []
timeseries_true.append(get_den(state_true))
timeseries_pred = []
timeseries_pred.append(get_den(state_pred))

state_pred = state_pred.reshape(1,-1)
state_real = np.real(state_pred)
state_imag = np.imag(state_pred)
state_pred = np.concatenate((state_real,state_imag),1)

for i in range(1000):
    
    state_true = np.matmul(prop,state_true)
    state_true_den = get_den(state_true)
    timeseries_true.append(state_true_den)

    state_pred = model.predict(state_pred)
    state_pred_packed = state_pred[0,:grid_size**2]+(state_pred[0,grid_size**2:]*1j)
    state_pred_den = get_den(state_pred_packed)
    state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
    state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
    state_pred_den = state_pred_den * (1/np.sum(state_pred_den))

    timeseries_pred.append(state_pred_den)
    
    print(i,get_den(np.sum(state_true*np.conj(state_pred_packed))))

timeseries_true = np.array(timeseries_true)    
timeseries_pred = np.array(timeseries_pred)

#%%

print(np.sum(timeseries_true))

#%%

xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
ylist = np.linspace(-0.75,0.75,grid_size)*1.8897
X, Y = np.meshgrid(xlist,ylist)

fig = plt.figure(dpi=100)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,timeseries_true[i].reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('True Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_true.gif', writer = 'imagemagick')

fig = plt.figure(dpi=100)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,timeseries_pred[i].reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Artificial Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_pred.gif', writer = 'imagemagick') 

fig = plt.figure(dpi=100)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,(np.abs(timeseries_pred[i]-timeseries_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Trajectory Error')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/2D_HBOND_err.gif', writer = 'imagemagick') 
    
#%%

plt.contourf(xlist,xlist,(nAQp.abs(timeseries_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])

#%%

i = 1000

fig = plt.figure(dpi=600)
plt.contourf(xlist,xlist,(np.abs(timeseries_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)

CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('x (bohr)')
plt.ylabel('y (bohr)') 
plt.title('Trajectory Error')
plt.show()

fig = plt.figure(dpi=600)
plt.contourf(xlist,xlist,(np.abs(timeseries_pred[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)

CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('x (bohr)')
plt.ylabel('y (bohr)') 
plt.title('Trajectory Error')
plt.show()

fig = plt.figure(dpi=600)
plt.contourf(xlist,xlist,(np.abs(timeseries_pred[i]-timeseries_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)

CS = plt.contour(X, Y, test_pot_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('x (bohr)')
plt.ylabel('y (bohr)') 
plt.title('Trajectory Error')
plt.show()


#%%

temp = 0

for j in range(100):

    state_0 = generate_state(basis,n_states)
    
    state_true = state_0
    state_pred = state_0
    
    timeseries_true = []
    timeseries_true.append(get_den(state_true))
    timeseries_pred = []
    timeseries_pred.append(get_den(state_pred))
    
    state_pred = state_pred.reshape(1,-1)
    state_real = np.real(state_pred)
    state_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_real,state_imag),1)
    
    for i in range(100):
        
        state_true = np.matmul(prop,state_true)
        state_true_den = get_den(state_true)
        timeseries_true.append(state_true_den)
    
        state_pred = model.predict(state_pred)
        state_pred_packed = state_pred[0,:grid_size**2]+(state_pred[0,grid_size**2:]*1j)
        state_pred_den = get_den(state_pred_packed)
        state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
        state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
        state_pred_den = state_pred_den * (1/np.sum(state_pred_den))
    
        timeseries_pred.append(state_pred_den)
        
        # print(i,get_den(np.sum(state_true*np.conj(state_pred_packed))))
        
    print(j,get_den(np.sum(state_true*np.conj(state_pred_packed))))
    
    temp = temp + get_den(np.sum(state_true*np.conj(state_pred_packed)))
    
    timeseries_true = np.array(timeseries_true)    
    timeseries_pred = np.array(timeseries_pred)   
    
print(temp/10)
