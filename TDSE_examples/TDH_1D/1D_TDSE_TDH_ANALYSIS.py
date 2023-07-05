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

from numba import jit
from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema
from scipy.interpolate import BSpline


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

#%%

sim_length = 500
n_states = 5
mass = 1836
grid_size = 64

dynamic_pot_list = []

start_pot = generate_pot()
test_pot_1 = start_pot

for q in range(10):

    test_pot_2 = generate_pot()
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,test_pot_1*627,xlist,test_pot_2*627)
    # plt.show()
    
    for i in range(50):
        step = i*np.pi/100
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

xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
ylist = np.linspace(-0.75,0.75,grid_size)*1.8897

X, Y = np.meshgrid(xlist,ylist)

timeseries_true = []
timeseries_state_true = []
timeseries_true.append(get_den(state_0))
timeseries_state_true.append(state_0)
timeseries_pos_true = []
timeseries_pos_true.append(np.sum(xlist*get_den(state_0)))

state = state_0
for i in range(sim_length):
    
    prop = generate_propagator(dynamic_pot_list[i])
    state_future = np.matmul(prop,state)
    state = state_future
    state_den = get_den(state)
    timeseries_true.append(state_den)
    timeseries_state_true.append(state)
    timeseries_pos_true.append(np.sum(xlist*get_den(state)))
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,dynamic_pot_list[i]*627,xlist,state_den*100)
    # plt.show()

timeseries_true = np.array(timeseries_true)

model = load_model('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_1D/saved_model')

timeseries_pred = []
timeseries_state_pred = []
timeseries_pos_pred = []
timeseries_pred.append(get_den(state_0))
timeseries_state_pred.append(state_0)
timeseries_pos_pred.append(np.sum(xlist*get_den(state_0)))

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
    state = state/np.sqrt(np.sum(state_den))
    state_packed = state_packed/np.sqrt(np.sum(state_den))
    state_den = state_den * (1/np.sum(state_den))
    timeseries_pred.append(state_den)
    timeseries_state_pred.append(state_packed)
    timeseries_pos_pred.append(np.sum(xlist*state_den))
    
    # plt.ylim(-1,50)
    # plt.plot(xlist,pot_temp[0]*627,xlist,timeseries_pred[i]*100)
    # plt.show()
    
timeseries_pred = np.array(timeseries_pred)

# timeseries_pred = []
# timeseries_state_pred = []
# timeseries_pos_pred = []
# timeseries_pred.append(get_den(state_0))
# timeseries_state_pred.append(state_0)
# timeseries_pos_pred.append(np.sum(xlist*get_den(state_0)))

# state = state_0
# for i in range(sim_length-1):
    
#     xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
#     hamiltonian = prop_1D(xlist,dynamic_pot_list[i],mass)
#     delta_t = 1000/24.18
#     prop_1 = expm(-1j*hamiltonian*delta_t*(1/4))
    
#     xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
#     hamiltonian = prop_1D(xlist,dynamic_pot_list[i+1],mass)
#     delta_t = 1000/24.18
#     prop_2 = expm(-1j*hamiltonian*delta_t*(1/2))
    
#     prop = np.matmul(prop_1,prop_2)
#     prop = np.matmul(prop,prop_1)
    
#     state_future = np.matmul(prop,state)
#     state = state_future
#     state_den = get_den(state)
#     timeseries_pred.append(state_den)
#     timeseries_state_pred.append(state)
#     timeseries_pos_pred.append(np.sum(xlist*get_den(state)))
    
#     # plt.ylim(-1,50)
#     # plt.plot(xlist,dynamic_pot_list[i]*627,xlist,state_den*100)
#     # plt.show()

# timeseries_true = np.array(timeseries_true)

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(sim_length):
# for i in range(5):
    
    tstep = i
    
    overlap = np.sum(timeseries_state_true[i]*(np.conj(timeseries_state_pred[i])))
    overlap_squared = np.real(overlap*np.conj(overlap))
    
    pos_error = np.abs(timeseries_pos_true[i]-timeseries_pos_pred[i])
    
    plt.text(0.75, 45, "Time Elapsed: "+str(str(int(tstep))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))
    plt.text(0.75, 41, r'$\langle\Psi^{true}|\Psi^{pred}\rangle$: '+"%.3f" % overlap_squared, color="k", fontsize=10, bbox=dict(facecolor='white'))
    plt.text(0.75, 37  , r'$|\langle x \rangle_{true}-\langle x \rangle_{pred}|$: '+"%.3f" % pos_error, color="k", fontsize=10, bbox=dict(facecolor='white'))
    
    plt.ylim(-5,50)
    # plt.plot(xlist,dynamic_pot_list[i]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b',xlist,(timeseries_pred[i]-timeseries_true[i])*100,'g')
    plt.plot(xlist,dynamic_pot_list[i]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b')
    test = BSpline(xlist,dynamic_pot_list[i]*627,1)
    plt.scatter(timeseries_pos_true[i], test(timeseries_pos_true[i]), marker='o', color='r')
    plt.scatter(timeseries_pos_pred[i], test(timeseries_pos_pred[i]), marker='o', color='b')
    
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
    
animation = camera.animate(interval=100)
animation.save('/Users/maximsecor/Desktop/TDSE_EXAMPLES/TDH_1D/1D_HBOND_true_slower_2.gif', writer = 'imagemagick')

    
    
    
    
    
    

