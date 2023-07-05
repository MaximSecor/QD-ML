#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:04:40 2020

@author: maximsecor
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
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

#%%
    
def get_den(state):
    return np.real(state*np.conj(state))

#%%
    
n_states = 100
mass = 1836

X0 = -1*np.random.random()+0.5

KX_1 = (2500*np.random.random()+1500)/(350*627)
KX_2 = (2500*np.random.random()+1500)/(350*627)
KY_1 = (2500*np.random.random()+1500)/(350*627)
KY_2 = (2500*np.random.random()+1500)/(350*627)
dE = np.abs(np.random.normal(0,1))/(627)

#%%

grid_size = 32

xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
ylist = np.linspace(-0.75,0.75,grid_size)*1.8897

X, Y = np.meshgrid(xlist,ylist)

potential_1 = 0.5*mass*(KX_1**2)*(X+0.5)**2 + 0.5*mass*(KY_1**2)*Y**2 
potential_2 = 0.5*mass*(KX_2**2)*(X+X0)**2 + 0.5*mass*(KY_2**2)*Y**2 + dE
couplings = np.full((grid_size,grid_size),10)*(1/627)

two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)

plt.contourf(xlist,ylist,ground_state_PES_1*627,levels = [0,10,25,50,75,100,200,300,400,500,600,700,800,900,1000])
plt.show()

# plt.contourf(xlist,ylist,ground_state_PES_2*627,levels = [0,10,25,50,75,100,200,300,400,500,600,700,800,900,1000])
# plt.show()

temp = fgh_2d(xlist,ground_state_PES_1,mass)
basis = temp[1][:,0:n_states].T

print(temp[0][:25])

plt.contourf(basis[0].reshape(grid_size,grid_size).T)
plt.show()   

# print(target_energies)

#%%

state_coeff = 2*np.random.random(n_states)-1
state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
state = np.matmul(state_coeff,basis)
state = state[0]

plt.contourf(state.reshape(grid_size,grid_size).T)
plt.show() 

#%%

# ground_state_PES_2 = ground_state_PES_1[:,::-1]
ground_state_PES_2 = ground_state_PES_1
hamiltonian = pot_2d(xlist,ground_state_PES_2,mass)
delta_t = 1000/24.18
prop = expm(-1j*hamiltonian*delta_t)

plt.contourf(xlist,ylist,ground_state_PES_2*627,levels = [0,10,25,50,75,100,200,300,400,500,600,700,800,900,1000])
plt.show()

#%%

for j in range(10):

    # phase = (2*np.random.random()-1)*np.pi
    # state_RP = state * np.exp(1j*phase)
    state_RP = state
    
    # non_stationaty_states_present.append(state_RP)
    state_future = np.matmul(prop,state_RP)
    # non_stationaty_states_future.append(state_future)
    state = state_future
    
    plt.contourf(get_den(state.reshape(grid_size,grid_size).T))
    plt.show()   

#%%

start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []

for i in range(100000):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state = np.matmul(state_coeff,basis)
    state = state[0]

    for j in range(1):
    
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

#%%

for i in range(8):
    # print(64*(2**i))
    NN_model.fit(train_tf, target_tf, epochs=4, batch_size=16*(2**i), verbose=2)

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_tf)
MAE = mean_absolute_error(target_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

grid_size_double = grid_size**2

def get_wave_input(feature):
    return  feature[:,0:(grid_size_double*1)]+(feature[:,(grid_size_double*1):(grid_size_double*2)]*np.complex(0,1))

def get_wave_output(feature):
    return  feature[:,0:(grid_size_double*1)]+(feature[:,(grid_size_double*1):(grid_size_double*2)]*np.complex(0,1))

#%%
    
input_wave = get_wave_input(val_X)
output_wave = get_wave_output(val_y)

#%%

i = 100
random_state = input_wave[i]
plt.contourf(get_den(random_state).reshape(grid_size,grid_size).T)

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

# eigen = 0
# random_state = basis[eigen]

# pos = 28
# random_state = np.zeros((64))
# random_state[pos] = 1

plot_den_true = []
plot_den_true.append(get_den(random_state))
teststep = random_state

for i in range(250):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    # plt.contourf(get_den(teststep).reshape(64,64))
    # plt.show()
  
plot_den_true = np.array(plot_den_true)

plot_den_pred = []
plot_den_pred.append(get_den(random_state))
teststep = random_state
teststep = teststep.reshape(1,-1)

teststep_real = np.real(teststep)
teststep_imag = np.imag(teststep)
teststep = np.concatenate((teststep_real,teststep_imag),1)

for i in range(250):
    teststep = NN_model.predict(teststep)
    teststep_f = teststep[0,:grid_size_double]+(teststep[0,grid_size_double:]*np.complex(0,1))
    test_den = get_den(teststep_f)
    teststep = teststep/np.sqrt(np.sum(test_den))
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.contourf(test_den.reshape(64,64))
    # plt.show()
    
# for i in range(250):
#     plt.ylim(-0.01,0.4)
#     plt.plot(xlist,xpot*0.01,xlist,plot_den_true[i],xlist,plot_den_pred[i])
#     plt.show()

plot_den_pred = np.array(plot_den_pred)

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,plot_den_true[i].reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, ground_state_PES_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('True Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/2D_HBOND_true.gif', writer = 'imagemagick')

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,plot_den_pred[i].reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, ground_state_PES_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Artificial Trajectory')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/2D_HBOND_pred.gif', writer = 'imagemagick') 


fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(250):
    
    tstep = i
    
    plt.contourf(xlist,xlist,(np.abs(plot_den_pred[i]-plot_den_true[i])).reshape(grid_size,grid_size).T,levels=[0.0,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1])
    plt.text(1.1, 1.25, str(str(int(tstep))+" fs"), color="w", fontsize=12)
    
    CS = plt.contour(X, Y, ground_state_PES_2*627,levels = [25,50,100,200,400,600], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel('x (bohr)')
    plt.ylabel('y (bohr)') 
    plt.title('Trajectory Error')
    
    camera.snap()
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/2D_HBOND_err.gif', writer = 'imagemagick') 



