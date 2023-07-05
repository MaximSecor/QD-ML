#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:14:22 2020

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

for i in range(10):
    pot_coeff = np.random.random(2)
    pot_coeff = pot_coeff * (1/np.sum(pot_coeff))
    print(pot_coeff)
    print(np.sum(pot_coeff))

#%%

start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []
acting_hamiltonian = []

for i in range(1):
    
    # Create double well
    
    grid_size = 64
    n_states = 5
    mass = 1836
    xlist = np.linspace(-1,1,grid_size)
    
    k1 = (1000*np.random.random()+2500)*(1/(627*350))
    k2 = (1000*np.random.random()+2500)*(1/(627*350))
    dist = 0.4*np.random.random()+0.1
    rel_en = (10*np.random.random()-5)*(1/(627))
    coupling = (5*np.random.random()+10)*np.full(grid_size,1)*(1/(627))
    
    state_a = 0.5*mass*(k1**2)*(xlist+dist)**2 - rel_en
    state_b = 0.5*mass*(k2**2)*(xlist-dist)**2 + rel_en
    
    react2state = np.array([[state_a,coupling],[coupling,state_b]]).T
    react_gs = np.linalg.eigh(react2state)[0].T
    react_gs = react_gs[0]
    react_gs = react_gs - np.min(react_gs)
    
    # plt.ylim(-1,50)
    # plt.plot(react_gs*627)
    # plt.show()
    
    # Solve TDSE
    
    react_soln = fgh(xlist*1.8897,react_gs,mass)
    react_wvfn = react_soln[1][:,0:n_states].T

    # Propogation
    
    k1 = (1000*np.random.random()+2500)*(1/(627*350))
    k2 = (1000*np.random.random()+2500)*(1/(627*350))
    dist = 0.4*np.random.random()+0.1
    rel_en = (10*np.random.random()-5)*(1/(627))
    coupling = (5*np.random.random()+10)*np.full(grid_size,1)*(1/(627))
    
    state_a = 0.5*mass*(k1**2)*(xlist+dist)**2 - rel_en
    state_b = 0.5*mass*(k2**2)*(xlist-dist)**2 + rel_en
    
    prod2state = np.array([[state_a,coupling],[coupling,state_b]]).T
    prod_gs = np.linalg.eigh(prod2state)[0].T
    prod_gs = prod_gs[0]
    prod_gs = prod_gs - np.min(prod_gs)
    
    hamiltonian = prop_ham(xlist*1.8897,prod_gs,mass)
    delta_t = 1000/24.18
    prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)
    
    for j in range(100):
        
        # Create random states
        
        state_coeff = 2*np.random.random(n_states)-1
        state_coeff = state_coeff * np.sqrt(1/np.sum(state_coeff**2))
        state = np.matmul(state_coeff,react_wvfn)
        
        for k in range(1000):
            
            # Propagate
            
            acting_hamiltonian.append(prod_gs)
            
            phase = (2*np.random.random()-1)*np.pi
            state_RP = state * np.exp(np.complex(0,1)*phase)
                    
            non_stationaty_states_present.append(state_RP)
            state_future = np.matmul(prop,state_RP)
            
            non_stationaty_states_future.append(state_future)
            state = state_future

end = time.time()
print(end-start)

# # state = react_wvfn[0]

# hamiltonian = prop_ham(xlist*1.8897,prod_gs,mass)
# delta_t = 1000/24.18
# prop = expm(-1j*hamiltonian*delta_t)

# for i in range(30):
    
#     state = np.matmul(prop,state)
    
#     plt.ylim(-1,50)
#     plt.plot(xlist,prod_gs*627,xlist,get_den(state)*100)
#     plt.show()

#%%

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)
present_ham = np.array(acting_hamiltonian)
present = np.concatenate((present_ham,present),1)

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

for i in range(1):
    NN_model.fit(train_tf, target_tf, epochs=4, batch_size=10000, verbose=2)

#%%
NN_model.save('/Users/maximsecor/Desktop/TDSE/saved_model_manypot')   
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

grid_size = 64

def get_pot_input(feature):
    return feature[:,0:grid_size]

def get_wave_input(feature):
    return  feature[:,grid_size:(grid_size*2)]+(feature[:,(grid_size*2):(grid_size*3)]*np.complex(0,1))

def get_wave_output(feature):
    return  feature[:,0:(grid_size*1)]+(feature[:,(grid_size*1):(grid_size*2)]*np.complex(0,1))

#%%
    
potentials = get_pot_input(val_X)
input_wave = get_wave_input(val_X)
output_wave = get_wave_output(val_y)

# potentials = get_pot_input(train_X)
# input_wave = get_wave_input(train_X)
# output_wave = get_wave_output(train_y)

#%%

i = 1120
random_state = input_wave[i]
random_pot = potentials[i]
    
plt.plot(xlist, random_pot, xlist, get_den(random_state))

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

# eigen = 1
# random_state = wvfns[eigen]

# pos = 28
# random_state = np.zeros((64))
# random_state[pos] = 1

hamiltonian = prop_ham(xlist*1.8897,random_pot,mass)
delta_t = 1000/24.18
prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)

plot_den_true = []
plot_den_true.append(get_den(random_state))
teststep = random_state

for i in range(100):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,get_den(teststep))
    # plt.show()

plot_den_pred = []
plot_den_pred.append(get_den(random_state))
teststep = random_state
teststep = teststep.reshape(1,-1)
random_pot = random_pot.reshape(1,-1)

teststep_real = np.real(teststep)
teststep_imag = np.imag(teststep)
teststep = np.concatenate((teststep_real,teststep_imag),1)
teststep = np.concatenate((random_pot,teststep),1)

for i in range(100):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    teststep = np.concatenate((random_pot,teststep),1)
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

for i in range(100):
    plt.ylim(-1,50)
    plt.plot(xlist, random_pot[0]*627, color = 'black')
    plt.plot(xlist, plot_den_true[i]*250, color = 'red')
    plt.plot(xlist, plot_den_pred[i]*250, color = 'blue')
    camera.snap()
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/test_ebc.gif', writer = 'imagemagick')   

#%%

k1 = (1000*np.random.random()+2500)*(1/(627*350))
k2 = (1000*np.random.random()+2500)*(1/(627*350))
dist = 0.2*np.random.random()+0.1
# dist = 0.2*np.random.random()+0.1
# rel_en = (10*np.random.random())*(1/(627))
rel_en = (1*np.random.random())*(1/(627))
coupling = (5*np.random.random()+10)*np.full(grid_size,1)*(1/(627))

state_a = 0.5*mass*(k1**2)*(xlist+dist)**2
state_b = 0.5*mass*(k2**2)*(xlist-dist)**2 + rel_en

react2state = np.array([[state_a,coupling],[coupling,state_b]]).T
react_gs = np.linalg.eigh(react2state)[0].T
react_gs = react_gs[0]
react_gs = react_gs - np.min(react_gs)

plt.ylim(-1,50)
plt.plot(react_gs*627)
plt.show()

# Solve TDSE

react_soln = fgh(xlist*1.8897,react_gs,mass)
react_wvfn = react_soln[1][:,0:n_states].T

prod_gs = react_gs[::-1]

plt.ylim(-1,50)
plt.plot(prod_gs*627)
plt.show()

random_state = react_wvfn[0]

hamiltonian = prop_ham(xlist*1.8897,react_gs,mass)
delta_t = 1000/24.18
prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)

potential_tracker = []
potential_tracker.append(react_gs)
plot_den_true = []
plot_den_true.append(get_den(random_state))
teststep = random_state

for i in range(50):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    potential_tracker.append(react_gs)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,get_den(teststep))
    # plt.show()

teststep_carryover = teststep

plot_den_pred = []
plot_den_pred.append(get_den(random_state))
teststep = random_state
teststep = teststep.reshape(1,-1)
random_pot = react_gs.reshape(1,-1)

teststep_real = np.real(teststep)
teststep_imag = np.imag(teststep)
teststep = np.concatenate((teststep_real,teststep_imag),1)
teststep = np.concatenate((random_pot,teststep),1)

for i in range(50):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    teststep = np.concatenate((random_pot,teststep),1)
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,test_den)
    # plt.show()

teststep_carryover_2 = teststep

hamiltonian = prop_ham(xlist*1.8897,prod_gs,mass)
delta_t = 1000/24.18
prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)

# plot_den_true = []
# plot_den_true.append(get_den(random_state))
teststep = teststep_carryover

for i in range(250):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    potential_tracker.append(prod_gs)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,get_den(teststep))
    # plt.show()

# plot_den_pred = []
# plot_den_pred.append(get_den(random_state))
# teststep = random_state
teststep = (teststep_carryover_2).reshape(1,-1)
random_pot = prod_gs.reshape(1,-1)

for i in range(250):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    teststep = np.concatenate((random_pot,teststep),1)
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,test_den)
    # plt.show()

fig = plt.figure(dpi=600)
camera = Camera(fig)

for i in range(300):
    plt.ylim(-1,50)
    plt.plot(xlist, potential_tracker[i]*627, color = 'black')
    plt.plot(xlist, plot_den_true[i]*250, color = 'red')
    plt.plot(xlist, plot_den_pred[i+1]*250, color = 'blue')
    camera.snap()
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/test_abc.gif', writer = 'imagemagick')   

#%%

k1 = (1000*np.random.random()+2500)*(1/(627*350))
k2 = (1000*np.random.random()+2500)*(1/(627*350))
dist = 0*np.random.random()+0.3
# rel_en = (1*np.random.random())*(1/(627))
rel_en = 0
coupling = (5*np.random.random()+10)*np.full(grid_size,1)*(1/(627))

state_a = 0.5*mass*(k1**2)*(xlist+dist)**2
state_b = 0.5*mass*(k2**2)*(xlist-dist)**2 + rel_en

react2state = np.array([[state_a,coupling],[coupling,state_b]]).T
react_gs = np.linalg.eigh(react2state)[0].T
react_gs = react_gs[0]
react_gs = react_gs - np.min(react_gs)

plt.ylim(-1,50)
plt.plot(react_gs*627)
plt.show()

# Solve TDSE

react_soln = fgh(xlist*1.8897,react_gs,mass)
react_wvfn = react_soln[1][:,0:n_states].T

prod_gs = react_gs[::-1]

plt.ylim(-1,50)
plt.plot(prod_gs*627)
plt.show()

random_state = react_wvfn[0]

hamiltonian = prop_ham(xlist*1.8897,react_gs,mass)
delta_t = 1000/24.18
prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)

potential_tracker = []
potential_tracker.append(react_gs)
plot_den_true = []
plot_den_true.append(get_den(random_state))
teststep = random_state

for i in range(10):
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    potential_tracker.append(react_gs)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,get_den(teststep))
    # plt.show()

teststep_carryover = teststep

plot_den_pred = []
plot_den_pred.append(get_den(random_state))
teststep = random_state
teststep = teststep.reshape(1,-1)
random_pot = react_gs.reshape(1,-1)

teststep_real = np.real(teststep)
teststep_imag = np.imag(teststep)
teststep = np.concatenate((teststep_real,teststep_imag),1)
teststep = np.concatenate((random_pot,teststep),1)

for i in range(10):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    teststep = np.concatenate((random_pot,teststep),1)
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,test_den)
    # plt.show()
    
teststep_carryover_2 = teststep


#%%

potential_acting = []

dist_0 = dist
rel_en_0 = rel_en

vx = 0.1
vy = -0.002
x = dist_0
y = rel_en_0
dt = 0.025

for i in range(1000):
    
    x = x + vx*dt
    vx = vx + (-1.3*(x-dist_0)*dt)
    
    y = y + vy*dt
    vy = vy + (-0.93*(y-rel_en_0)*dt)
    
    print(y)
    
    dist = x
    rel_en = y
    coupling = coupling
    
    state_a = 0.5*mass*(k1**2)*(xlist+dist)**2 - rel_en
    state_b = 0.5*mass*(k2**2)*(xlist-dist)**2 + rel_en
    
    react2state = np.array([[state_a,coupling],[coupling,state_b]]).T
    react_gs = np.linalg.eigh(react2state)[0].T
    react_gs = react_gs[0]
    react_gs = react_gs - np.min(react_gs)
    
    potential_acting.append(react_gs)
    
    plt.ylim(-10,50)
    plt.plot(xlist,react_gs*627)
    plt.show
    
potential_acting = np.array(potential_acting)
    
#%%

teststep = teststep_carryover

for i in range(1000):
    
    prod_gs = potential_acting[i]
    
    hamiltonian = prop_ham(xlist*1.8897,prod_gs,mass)
    delta_t = 1000/24.18
    prop = expm((-1)*np.complex(0,1)*hamiltonian*delta_t)
    
    teststep = np.matmul(prop,teststep)
    plot_den_true.append(get_den(teststep))
    potential_tracker.append(prod_gs)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,prod_gs,xlist,get_den(teststep))
    # plt.show()

#%%

# plot_den_pred = []
# plot_den_pred.append(get_den(random_state))
# teststep = random_state
teststep = (teststep_carryover_2).reshape(1,-1)
random_pot = prod_gs.reshape(1,-1)

for i in range(1000):
    teststep = NN_model.predict(teststep)
    teststep_128 = teststep[0,:grid_size]+(teststep[0,grid_size:]*np.complex(0,1))
    test_den = get_den(teststep_128)
    teststep = teststep/np.sum(test_den)
    prod_gs = potential_acting[i]
    random_pot = prod_gs.reshape(1,-1)
    teststep = np.concatenate((random_pot,teststep),1)
    # print(np.sum(test_den))
    test_den = test_den * (1/np.sum(test_den))
    plot_den_pred.append(test_den)
    # plt.ylim(-0.01,0.4)
    # plt.plot(xlist,test_den)
    # plt.show

#%%
   
for i in range(250):
    plt.ylim(-1,50)
    plt.plot(xlist, potential_tracker[i*4]*627, color = 'black')
    plt.plot(xlist, plot_den_true[i*4]*250, color = 'red')
    plt.plot(xlist, plot_den_pred[i*4+1]*250, color = 'blue')
    plt.show()
    
#%%

fig = plt.figure(dpi=600)
camera = Camera(fig)

for i in range(1000):
    plt.ylim(-1,50)
    plt.plot(xlist, potential_tracker[i]*627, color = 'black')
    plt.plot(xlist, plot_den_true[i]*250, color = 'red')
    plt.plot(xlist, plot_den_pred[i+1]*250, color = 'blue')
    camera.snap()
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/test_wow_4.gif', writer = 'imagemagick')   

#%%





