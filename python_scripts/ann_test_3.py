#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:54:05 2021

@author: maximsecor
"""


import numpy as np
from numba import jit
import time
import os
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera

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
        
    potential_trajectory = np.array(potential_trajectory)
    
    return potential_trajectory

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
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator(xlist,potential,dt):
    
    hamiltonian = prop_1D(xlist,potential,1836)
    delta_t = dt/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

grid_size = 128
X_eq = 0
R_eq = 1

w_p1 = 1500 
w_p2 = 1500

couplings = 10

test = []
for i in range(100):
    
    R_eq = 2 - 0.01*i
    test.append(generate_pot_test(grid_size, X_eq, R_eq, w_p1, w_p2, couplings)[0])
    
test = np.array(test)
plt.contourf(test)

#%%

grid_size = 128
X_eq = 2
R_eq = 1

w_p1 = 1500 
w_p2 = 1500

couplings = 10

test = []
for i in range(100):
    
    R_eq = 2 - 0.01*i
    test.append(generate_pot_test(grid_size, X_eq, R_eq, w_p1, w_p2, couplings)[0])
    
test = np.array(test)
plt.contourf(test)

#%%

length = 1000
test = generate_pot_TD(length)

#%%

plt.contourf(test.T)

#%%

print(np.linspace(0,99,100)[::10])
print(len(np.linspace(0,99,100)))

#%%

grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 1
potential_traj_length = 20
pot_gap_step = 2

acting_potential_traj = []
non_stationaty_states_present = []
non_stationaty_states_future = []

start = time.time()

for i in range(10000):
    
    potential_traj = generate_pot_TD(potential_traj_length)
    basis = generate_basis(xlist,potential_traj[0],n_states)
    state = generate_state(basis,n_states)
    phase = (2*np.random.random()-1)*np.pi
    state_RP = state * np.exp(1j*phase)
    
    prop = generate_propagator(xlist,potential_traj[0],1000*np.random.random())
    state_RP = np.matmul(prop,state_RP)
    non_stationaty_states_present.append(state_RP)
    
    for j in range(potential_traj_length):
        prop = generate_propagator(xlist,potential_traj[j],1000)
        state_RP = np.matmul(prop,state_RP)
    non_stationaty_states_future.append(state_RP)
    
    acting_potential_traj.append(np.concatenate((potential_traj[::pot_gap_step].reshape(grid_size*(int(potential_traj_length/pot_gap_step))),potential_traj[j]),0))

acting_potential_traj = np.array(acting_potential_traj)
non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)

end = time.time()
print("Datagen Time: ", end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)
present = np.concatenate((acting_potential_traj,present),1)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

features = present
target = future

dense_layers = 3
dense_nodes = 512
learning_rates = 0.001

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
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('Training Time: ',(end-start))

#%%

start = time.time()

total_time = 1000
n_states = 1
ANN_step = int(total_time/potential_traj_length)
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897

trials = 100
overlap_error_total = np.zeros((ANN_step))

for q in range(trials):
    
    print(q)
    
    potential = generate_pot_TD(total_time+1)
    basis = generate_basis(xlist,potential[0],n_states)
    state = generate_state(basis,n_states)

    state_pred = state
    state_true = state
    
    state_pred = state_pred.reshape(1,-1)
    state_pred_real = np.real(state_pred)
    state_pred_imag = np.imag(state_pred)
    state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
    state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
    
    test_viz = []
    test_viz_true = []
    
    overlap_error = np.zeros((ANN_step))
    
    for i in range(ANN_step):
        
        test_viz.append(state_pred_packed)
        test_viz_true.append(state_true)
        
        ann_in = np.concatenate((potential[potential_traj_length*i],potential[potential_traj_length*i+pot_gap_step]),0)
        
        for q in range(int(potential_traj_length/pot_gap_step)-1):
            ann_in = np.concatenate((ann_in,potential[potential_traj_length*i+(2+q)*pot_gap_step]),0)
        
        ann_in = np.concatenate((ann_in,state_pred.reshape(64)),0)
        ann_in = ann_in.reshape(-1,1).T
        state_pred = model.predict(ann_in)
        
        state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
        state_pred_den = get_den(state_pred_packed)
        state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
        state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
        state_pred_den = state_pred_den * (1/np.sum(state_pred_den))
        
        for t in range(potential_traj_length):
            prop = generate_propagator(xlist,potential[potential_traj_length*i+t],1000)
            state_true = np.matmul(prop,state_true)
        
        # plt.ylim(-1,50)
        # # plt.plot(xlist/1.8897,potential[potential_traj_length*i]*627)
        # # plt.plot(xlist/1.8897,potential[potential_traj_length*i+pot_gap_step]*627)
        # plt.plot(xlist/1.8897,potential[potential_traj_length*i+10*pot_gap_step]*627)
        # plt.plot(xlist/1.8897,get_den(state_true)*50)
        # plt.plot(xlist/1.8897,get_den(state_pred_packed)*50)
        # plt.show()
        
        overlap_error[i] = overlap_error[i] + np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true))))
        
    overlap_error_total += overlap_error
    # plt.plot(overlap_error)
    # plt.show()

plt.plot(overlap_error_total/trials)
plt.show()

end = time.time()
print('Training Time: ',(end-start))
    
#%%

potential = generate_pot_TD(total_time+1)
basis = generate_basis(xlist,potential[0],n_states)
state = generate_state(basis,n_states)

state_pred = state
state_true = state

state_pred = state_pred.reshape(1,-1)
state_pred_real = np.real(state_pred)
state_pred_imag = np.imag(state_pred)
state_pred = np.concatenate((state_pred_real,state_pred_imag),1)
state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)

test_viz = []
test_viz_true = []

for i in range(ANN_step):
    
    test_viz.append(state_pred_packed)
    test_viz_true.append(state_true)

    ann_in = np.concatenate((potential[0].reshape(-1,1).T,state_pred),1)
    state_pred = model.predict(ann_in)
    state_pred_packed = state_pred[0,:grid_size]+(state_pred[0,grid_size:]*1j)
    state_pred_den = get_den(state_pred_packed)
    state_pred = state_pred/np.sqrt(np.sum(state_pred_den))
    state_pred_packed = state_pred_packed/np.sqrt(np.sum(state_pred_den))
    state_pred_den = state_pred_den * (1/np.sum(state_pred_den))

    prop = generate_propagator(xlist,potential[i],1000)
    state_true = np.matmul(prop,state_true)
    overlap_error[i] = overlap_error[i] + np.sqrt(get_den(np.sum(state_pred_packed*np.conj(state_true))))
    
plt.plot(overlap_error)
plt.show()

plt.ylim(-1,50)
plt.plot(xlist/1.8897,potential[i]*627)
plt.plot(xlist/1.8897,get_den(state_true)*50)
plt.show()

#%%

fig = plt.figure(dpi=100)
camera = Camera(fig)

for i in range(500):
    
    plt.text(0.75, 45, "Time Elapsed: "+str(str(int(i))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))
    plt.ylim(-5,50)
    # plt.plot(xlist,dynamic_pot_list[i]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b',xlist,(timeseries_pred[i]-timeseries_true[i])*100,'g')
    plt.plot(xlist,get_den(test_viz[i])*100,'b',xlist,get_den(test_viz_true[i])*100,'r',xlist,potential[i]*627,'k')
    # plt.plot(xlist,test[i]*627,'k')
    
    plt.xlabel('x (bohr)')
    plt.ylabel('Energy (kcal/mol)')
    # plt.title('Constantly Changing Hamiltonian')
    
    plt.rc('font', family='Helvetica')
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    camera.snap()
    
animation = camera.animate(interval=20)
animation.save('/Users/maximsecor/Desktop/potentials.gif', writer = 'imagemagick')