#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:11:30 2021

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
from scipy.interpolate import interp1d

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

def generate_pot_TD_2(length,K_p1,K_p2,R_freq,R_eq,R_0,X_eq,lam,phase_1,phase_2):

    dt_fs = 1000/24.188
    
    grid_size = 32
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    mass = 1836
    
    f0 = 55.734
    mu = 0.2652
    X_freq = np.sqrt(f0/mu)*0.001
    
    # phase_1 = 2*np.pi*np.random.random()
    # phase_2 = 2*np.pi*np.random.random()

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

def generate_pot_test_2(grid_size,X_eq,R_eq,w_p1,w_p2,couplings):
    
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    mass = 1836
    K_p1 = mass*((w_p1)/(350*627))**2
    K_p2 = mass*((w_p2)/(350*627))**2
    
    potential_1 = 0.5*K_p1*(xlist-(0.5*R_eq))**2
    potential_2 = 0.5*K_p2*(xlist+(0.5*R_eq))**2 + X_eq*(1/627)
    couplings = couplings*np.exp(-(xlist**2)/2)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    potential_gs = np.linalg.eigh(two_state.T)[0].T[0]
    potential_es = np.linalg.eigh(two_state.T)[0].T[1]
    
    return potential_gs,potential_es


def generate_basis(xlist,potential,n_states):
    
    temp = fgh_1D(xlist,potential,1836)
    basis = temp[1][:,0:n_states].T
    
    return basis


def generate_spectra(xlist,potential,n_states):
    
    temp = fgh_1D(xlist,potential,1836)
    spec = temp[0][0:n_states]
    
    return spec

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
    
grid_size = 1024
X_eq = [0,0,10,10,10,10,20,20]
R_eq = [0,0.5,0,0.5,0.75,0.75,1,1]
w_p1 = [1500,1500,1500,1500,4000,4000,4000,4000]
w_p2 = [4000,1500,4000,1500,4000,1500,4000,1500]
couplings = 10

xlist = np.linspace(-1.5,1.5,grid_size)

for i in range(4):

    test_plot = generate_pot_test(grid_size,X_eq[i],R_eq[i]*1.8897,w_p1[i],w_p2[i],couplings)[0]
    test_plot = test_plot-min(test_plot)
       
    plt.ylim(-1,50)
    plt.plot(xlist,test_plot*627)
    plt.show()

#%%
   
grid_size = 1024
X_eq = [0,    0,    20,   20,   10,   10,   20,    0,    0,    0,    0,    0,    0,    0,    10,   5,    -10,  0]
R_eq = [0,    0,    0.75, -0.5, 0.25, 0.5,  -0.25, 0,    0.75, 0.3,  1,    1.5,  2,    1,    0.5,  0.5,  1.5,  0.51]
w_p1 = [1500, 4000, 1500, 1500, 4000, 1500, 4000,  1500, 1500, 4000, 1500, 4000, 4000, 1500, 4000, 1500, 1500, 1500]
w_p2 = [1500, 4000, 1500, 4000, 4000, 1500, 4000,  4000, 1500, 4000, 1500, 4000, 1500, 4000, 1500, 4000, 1500, 1500]
couplings = 10

# grid_size = 1024
# X_eq = 40*np.random.random(18)-20
# R_eq = np.random.random(18)
# w_p1 = 2500*np.random.random(18)+1500
# w_p2 = 2500*np.random.random(18)+1500
# couplings = 10

xlist = np.linspace(-1.5,1.5,grid_size)
    
fig11 = plt.figure(figsize=(18, 24), constrained_layout=False)
outer_grid = fig11.add_gridspec(6, 3, wspace=0.0, hspace=0.0)

plot_exc = [1,2,4,5,7,8,10,11,13,14]
plot_exc_2 = [16,17]
plot_exc_3 = [3,6,9,12,15]

SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for i in range(18):
    
    test_plot = generate_pot_test(grid_size,X_eq[i],R_eq[i]*1.8897,w_p1[i],w_p2[i],couplings)[0]
    test_plot = test_plot-min(test_plot)
    
    print(i,np.max(test_plot[400:700])*627)
    
    # plt.ylim(-1,50)
    
    if i in plot_exc:

        plt.ylim(-1,50)
        ax = fig11.add_subplot(outer_grid[i])
        ax.plot(xlist,test_plot*627)
        ax.set_xticks([])
        ax.set_yticks([])
        fig11.add_subplot(ax)
        
    elif i in plot_exc_2:
    
        plt.ylim(-1,50)
        ax = fig11.add_subplot(outer_grid[i])
        plt.ylim(-1,50)
        ax.plot(xlist,test_plot*627)
        ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
        ax.set_yticks([])
        fig11.add_subplot(ax)
        
    elif i in plot_exc_3:

        plt.ylim(-1,50)
        ax = fig11.add_subplot(outer_grid[i])
        ax.plot(xlist,test_plot*627)
        ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
        ax.set_yticks([0,10,20,30,40])
        fig11.add_subplot(ax)
        
    else:  
        
        # plt.ylim(-1,50)
        ax = fig11.add_subplot(outer_grid[i])
        ax.plot(xlist,test_plot*627)
        ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
        ax.set_yticks([0,10,20,30,40,50])
        fig11.add_subplot(ax)
        

all_axes = fig11.get_axes()

plt.show()


#%%

plt.plot(xlist,generate_pot_test_2(grid_size,0,1.8897*0.8,1500,1500,10)[2])

#%%

temp_name = '/Users/maximsecor/Desktop/saved_model'
model = load_model(temp_name)

#%%

grid_size = 32
X_eq = 0
R_eq = 0.5*1.8897
w_p1 = 1500
w_p2 = 1500
couplings = 10

xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
potential = generate_pot_test(grid_size,X_eq,R_eq,w_p1,w_p2,couplings)[0]
# potential = generate_pot_test(grid_size,5,1.8897*0,1500,1500,10)[0]
# potential = generate_pot(grid_size)
potential = potential - min(potential)
# plt.ylim(-1,50)
# plt.plot(xlist,potential*627)

length = 350
grid_size = 32

test_chk = potential
test_pot = []
for i in range(length):
    test_pot.append(test_chk)
test_pot = np.array(test_pot)

plt.ylim(-1,50)
plt.plot(test_pot[0]*627)

#%%

test_pot = potential

xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 5

test_basis = generate_basis(xlist,potential,n_states)
test_state = generate_state(test_basis,n_states)
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

spline_pot = interp1d(xlist, potential, kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_out_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

plt.ylim(-1,50)
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.rc('font', family='Helvetica')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

i = 0
plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)      
plt.show()

#%%


n_states = 1
overlap = 0
trial = 0

while overlap < 0.9:
    
    i = 0
    
    trial = trial + 1
    
    test_basis = generate_basis(xlist,potential,n_states)
    test_state = generate_state(test_basis,n_states)
    test_out = np.concatenate((np.real(test_state),np.imag(test_state)))
    
    xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
    test_out_viz = test_out[:32]+test_out[32:]*1j
    test_out_real = test_out_viz

    test_real = test_out_viz
    
    spline_pot = interp1d(xlist, potential, kind='cubic')
    spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
    spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
    
    fig_pot = spline_pot(xlist_res)*627
    # fig_pot[np.where(fig_pot<0)] = 0
    
    fig_ann = spline_ann(xlist_res)*100
    fig_ann[np.where(fig_ann<0)] = 0
    
    fig_real = spline_prp(xlist_res)*100
    fig_real[np.where(fig_real<0)] = 0
    
    i = 0
    overlap_temp = 1

    start = time.time()

    while overlap_temp > 0.9 and i < 500:
        
        # print(i,overlap_temp)
        
        i = i + 1
        
        test_prop = generate_propagator(xlist,potential,1000)
        test_in = np.concatenate((potential,test_out)).reshape(1,-1)
        test_out = model.predict(test_in)
        test_out_viz = test_out[0,:32]+test_out[0,32:]*1j
    
        test_real = np.matmul(test_prop,test_real)
    
        test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
        test_out = test_out[0]
        test_out_viz = test_out[:32]+test_out[32:]*1j
        
        overlap_temp = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    
    spline_pot = interp1d(xlist, potential, kind='cubic')
    spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
    spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
    
    fig_pot = spline_pot(xlist_res)*627
    # fig_pot[np.where(fig_pot<0)] = 0
    
    fig_ann = spline_ann(xlist_res)*100
    fig_ann[np.where(fig_ann<0)] = 0
    
    fig_real = spline_prp(xlist_res)*100
    fig_real[np.where(fig_real<0)] = 0
    
    overlap = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    end = time.time()
    print(trial, i, end-start, overlap)

#%%
    
ham = prop_1D(xlist,potential,1836)
    
fig11 = plt.figure(figsize=(18, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 3, wspace=0.0, hspace=0.0)
plot_exc = [1,2]
plot_exc_2 = [4,5]    

plot_idx = 0
    
i = 0

overlap_list = []
pos_error_list = []
nrg_error_list = []

pos_pred = []
nrg_pred = []

pos_real = []
nrg_real = []
  
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

test_real = test_out_viz

spline_pot = interp1d(xlist, potential, kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

# plt.ylim(-1,50)

ax = fig11.add_subplot(outer_grid[plot_idx])
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.ylim(-1,50)
ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax.set_yticks([0,10,20,30,40,50])
fig11.add_subplot(ax)

V_SMALL_SIZE = 18
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)

plot_idx = plot_idx + 1

for i in range(500):
    
    start = time.time()
    
    i = i + 1
    
    test_prop = generate_propagator(xlist,potential,1000)
    test_in = np.concatenate((potential,test_out)).reshape(1,-1)
    test_out = model.predict(test_in)
    test_out_viz = test_out[0,:32]+test_out[0,32:]*1j

    test_real = np.matmul(test_prop,test_real)

    overlap = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    pos_exp_real = np.sum(get_den(test_real)*xlist)
    pos_exp_pred = np.sum(get_den(test_out_viz)*xlist)
    
    pos_nrg_real = np.sqrt(get_den(np.matmul(np.matmul(ham,test_real),np.conj(test_real))))*627
    pos_nrg_pred = np.sqrt(get_den(np.matmul(np.matmul(ham,test_out_viz),np.conj(test_out_viz))))*627
    
    pos_real.append(pos_exp_real)
    nrg_real.append(pos_nrg_real)
    
    pos_pred.append(pos_exp_pred)
    nrg_pred.append(pos_nrg_pred)
    
    pos_error = pos_exp_pred-pos_exp_real
    nrg_error = pos_nrg_pred-pos_nrg_real

    print('\n')
    print('Step: ', i)
    print('Overlap: ', overlap)
    print('Position Expectation Error: ',pos_error)
    print('Energy Expectation Error: ',nrg_error)
    
    overlap_list.append(overlap)
    pos_error_list.append(pos_error)
    nrg_error_list.append(nrg_error)
    
    if i%100==0:
        
        spline_pot = interp1d(xlist, potential, kind='cubic')
        spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
        spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
        
        fig_pot = spline_pot(xlist_res)*627
        # fig_pot[np.where(fig_pot<0)] = 0
        
        fig_ann = spline_ann(xlist_res)*100
        # fig_ann = fig_ann - min(fig_ann)
        fig_ann[np.where(fig_ann<0)] = 0
        
        fig_real = spline_prp(xlist_res)*100
        # fig_real = fig_real - min(fig_real)
        fig_real[np.where(fig_real<0)] = 0
        
        V_SMALL_SIZE = 18
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 24
        
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        plt.ylim(-1,50)
        # plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)
        
        plot_exc = [1,2]
        plot_exc_2 = [4,5]    

        if plot_idx in plot_exc:
    
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        elif plot_idx in plot_exc_2:
            
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        else:  
        
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([0,10,20,30,40])
            fig11.add_subplot(ax)
            
        plot_idx = plot_idx + 1
        
        plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)
        
    test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
    test_out = test_out[0]

overlap_list = np.array(overlap_list)
pos_error_list = np.array(pos_error_list)
nrg_error_list = np.array(nrg_error_list)

pos_pred = np.array(pos_pred)
pos_real = np.array(pos_real)
nrg_pred = np.array(nrg_pred)
nrg_real = np.array(nrg_real)

plt.ylim(-1,50)
plt.show()

#%%

plt.plot(pos_pred/1.889,"r")
plt.plot(pos_real/1.889,"b--")
plt.show()

plt.plot(nrg_pred,"r")
plt.plot(nrg_real,"b--")
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_pred/1.889,"r")
ax.plot(pos_real/1.889,"b--")
# ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([])
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_pred,"r")
ax.plot(nrg_real,"b--")
ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([])
fig11.add_subplot(ax)

plt.show()
#%%

plt.plot(overlap_list)
plt.show()

plt.plot(pos_error_list/1.889)
plt.show()

plt.plot(nrg_error_list)
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 12), constrained_layout=False)
outer_grid = fig11.add_gridspec(3, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_error_list/1.889,'k')
ax.set_xticks([])
# ax.set_yticks([])
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_error_list,'k')
ax.set_xticks([])
# ax.set_yticks([])
fig11.add_subplot(ax)

i = 2
ax = fig11.add_subplot(outer_grid[i])
ax.plot(overlap_list,'k')
ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([])
fig11.add_subplot(ax)


plt.show()

#%%
    
mass = 1836
length = 1500
K_p1 = mass*(1500/(350*627))**2
K_p2 = mass*(1500/(350*627))**2
R_freq = ((200)/(627*350))
R_eq = (1)*1.8897
R_0 = 2*R_eq
X_eq = 0
lam = 0
phase_1 = np.pi*0
phase_2 = np.pi*0

n_states = 1

test_pot = generate_pot_TD_2(length,K_p1,K_p2,R_freq,R_eq,R_0,X_eq,lam,phase_1,phase_2)

xlist = np.linspace(-1.5,1.5,grid_size)*1.8897

test_basis = generate_basis(xlist,test_pot[0],n_states)
test_state = generate_state(test_basis,n_states)
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

spline_pot = interp1d(xlist, test_pot[0], kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_out_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

plt.ylim(-1,50)
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.rc('font', family='Helvetica')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)      

plt.show()

test_real = test_out_viz

overlap = 0
trial = 0

while overlap < 0.9:
    
    i = 0
    
    trial = trial + 1
    # print(trial)
    
    test_basis = generate_basis(xlist,test_pot[i],n_states)
    test_state = generate_state(test_basis,n_states)
    test_out = np.concatenate((np.real(test_state),np.imag(test_state)))
    
    xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
    test_out_viz = test_out[:32]+test_out[32:]*1j
    test_out_real = test_out_viz

    test_real = test_out_viz
    
    spline_pot = interp1d(xlist, potential, kind='cubic')
    spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
    spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
    
    fig_pot = spline_pot(xlist_res)*627
    # fig_pot[np.where(fig_pot<0)] = 0
    
    fig_ann = spline_ann(xlist_res)*100
    fig_ann[np.where(fig_ann<0)] = 0
    
    fig_real = spline_prp(xlist_res)*100
    fig_real[np.where(fig_real<0)] = 0

    i = 0
    overlap_temp = 1

    start = time.time()

    while overlap_temp > 0.9 and i < 500:
        
        i = i + 1
        
        test_prop = generate_propagator(xlist,test_pot[i],1000)
        test_in = np.concatenate((test_pot[i],test_out)).reshape(1,-1)
        test_out = model.predict(test_in)
        test_out_viz = test_out[0,:32]+test_out[0,32:]*1j
    
        test_real = np.matmul(test_prop,test_real)
    
        test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
        test_out = test_out[0]
        test_out_viz = test_out[:32]+test_out[32:]*1j
        
        overlap_temp = get_den(np.sum(test_out_viz*np.conj(test_real)))

    overlap = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    end = time.time()
    print(trial, i, end-start, overlap)
    
#%%
    
plot_idx = 0
i = 0

    
fig11 = plt.figure(figsize=(18, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 3, wspace=0.0, hspace=0.0)
plot_exc = [1,2]
plot_exc_2 = [4,5]  

overlap_list = []
pos_error_list = []
nrg_error_list = []

pos_pred = []
nrg_pred = []

pos_real = []
nrg_real = []
  
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

test_real = test_out_viz

spline_pot = interp1d(xlist, test_pot[i], kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

# plt.ylim(-1,50)

ax = fig11.add_subplot(outer_grid[plot_idx])
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.ylim(-1,50)
ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax.set_yticks([0,10,20,30,40,50])
fig11.add_subplot(ax)

V_SMALL_SIZE = 18
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)

plot_idx = plot_idx + 1



for i in range(500):
    
    start = time.time()
    
    i = i + 1
    
    potential = test_pot[i]
    
    ham = prop_1D(xlist,potential,1836)
    
    test_prop = generate_propagator(xlist,potential,1000)
    test_in = np.concatenate((potential,test_out)).reshape(1,-1)
    test_out = model.predict(test_in)
    test_out_viz = test_out[0,:32]+test_out[0,32:]*1j

    test_real = np.matmul(test_prop,test_real)

    overlap = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    pos_exp_real = np.sum(get_den(test_real)*xlist)
    pos_exp_pred = np.sum(get_den(test_out_viz)*xlist)
    
    pos_nrg_real = np.sqrt(get_den(np.matmul(np.matmul(ham,test_real),np.conj(test_real))))*627
    pos_nrg_pred = np.sqrt(get_den(np.matmul(np.matmul(ham,test_out_viz),np.conj(test_out_viz))))*627
    
    pos_real.append(pos_exp_real)
    nrg_real.append(pos_nrg_real)
    
    pos_pred.append(pos_exp_pred)
    nrg_pred.append(pos_nrg_pred)
    
    pos_error = pos_exp_pred-pos_exp_real
    nrg_error = pos_nrg_pred-pos_nrg_real

    print('\n')
    print('Step: ', i)
    print('Overlap: ', overlap)
    print('Position Expectation Error: ',pos_error)
    print('Energy Expectation Error: ',nrg_error)
    
    overlap_list.append(overlap)
    pos_error_list.append(pos_error)
    nrg_error_list.append(nrg_error)
    
    if i%100==0:
        
        spline_pot = interp1d(xlist, potential, kind='cubic')
        spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
        spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
        
        fig_pot = spline_pot(xlist_res)*627
        # fig_pot[np.where(fig_pot<0)] = 0
        
        fig_ann = spline_ann(xlist_res)*100
        # fig_ann = fig_ann - min(fig_ann)
        fig_ann[np.where(fig_ann<0)] = 0
        
        fig_real = spline_prp(xlist_res)*100
        # fig_real = fig_real - min(fig_real)
        fig_real[np.where(fig_real<0)] = 0
        
        V_SMALL_SIZE = 18
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 24
        
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        plt.ylim(-1,50)
        # plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)
        
        plot_exc = [1,2]
        plot_exc_2 = [4,5]    

        if plot_idx in plot_exc:
    
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        elif plot_idx in plot_exc_2:
            
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        else:  
        
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([0,10,20,30,40])
            fig11.add_subplot(ax)
        
        plt.ylim(-1,50)
        plot_idx = plot_idx + 1
        
        plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)
        
    test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
    test_out = test_out[0]

overlap_list = np.array(overlap_list)
pos_error_list = np.array(pos_error_list)
nrg_error_list = np.array(nrg_error_list)

pos_pred = np.array(pos_pred)
pos_real = np.array(pos_real)
nrg_pred = np.array(nrg_pred)
nrg_real = np.array(nrg_real)

#%%

plt.plot(pos_pred/1.889)
plt.plot(pos_real/1.889)
plt.show()

plt.plot(nrg_pred)
plt.plot(nrg_real)
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_pred/1.889,"r")
ax.plot(pos_real/1.889,"b--")
# ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([])
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_pred,"r")
ax.plot(nrg_real,"b--")
ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([4,8,12,16])
fig11.add_subplot(ax)

plt.show()

#%%

plt.plot(overlap_list)
plt.show()

plt.plot(pos_error_list/1.889)
plt.show()

plt.plot(nrg_error_list)
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 12), constrained_layout=False)
outer_grid = fig11.add_gridspec(3, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_error_list/1.889,'k')
ax.set_xticks([])
# ax.set_yticks([])
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_error_list,'k')
ax.set_xticks([])
# ax.set_yticks([])
fig11.add_subplot(ax)

i = 2
ax = fig11.add_subplot(outer_grid[i])
ax.plot(overlap_list,'k')
ax.set_xticks([0,100,200,300,400,500])
# ax.set_yticks([])
fig11.add_subplot(ax)


plt.show()

#%%

grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
potential = generate_pot_test(grid_size,10,1.8897*1.5,4000,4000,10)[0]
potential = potential - min(potential)
plt.ylim(-1,50)
plt.plot(xlist,potential*627)
    
#%%
    
grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
# potential = generate_pot(grid_size)

n_states = 5
basis = generate_basis(xlist,potential,n_states)
prop = generate_propagator(xlist,potential,1000)

non_stationaty_states_present = []
non_stationaty_states_future = []

start = time.time()
for i in range(100000):

    state = generate_state(basis,n_states)

    for j in range(1):

        state_RP = state
        non_stationaty_states_present.append(state_RP)
        state_future = np.matmul(prop,state_RP)
        non_stationaty_states_future.append(state_future)
        state = state_future

non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)

end = time.time()
print("Datagen Time: ", end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

features = present
target = future

dense_layers = 0
dense_nodes = 128
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
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('Training Time: ',(end-start))

# print("Complex Error")

# predictions_train = model.predict(train_X_tf)
# MAE = mean_absolute_error(train_y_tf, predictions_train)
# print('Training Set Error =', MAE)

# predictions_val = model.predict(val_X_tf)
# MAE = mean_absolute_error(val_y_tf, predictions_val)
# print('Cross-Validation Set Error =', MAE)

# print("\nDensity Error")

# predictions_train = model.predict(train_X_tf)
# MAE = mean_absolute_error(get_den(train_y_tf), get_den(predictions_train))
# print('Training Set Error =', MAE)

# predictions_val = model.predict(val_X_tf)
# MAE = mean_absolute_error(get_den(val_y_tf), get_den(predictions_val))
# print('Cross-Validation Set Error =', MAE)

#%%


xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 5

test_basis = generate_basis(xlist,potential,n_states)
test_state = generate_state(test_basis,n_states)
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

ham = prop_1D(xlist,potential,1836)
test_prop = generate_propagator(xlist,potential,1000)
    
fig11 = plt.figure(figsize=(18, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 3, wspace=0.0, hspace=0.0)
plot_exc = [1,2]
plot_exc_2 = [4,5]    

plot_idx = 0
    
i = 0

overlap_list = []
pos_error_list = []
nrg_error_list = []

pos_pred = []
nrg_pred = []

pos_real = []
nrg_real = []
  
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

test_real = test_out_viz

spline_pot = interp1d(xlist, potential, kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

# plt.ylim(-1,50)

ax = fig11.add_subplot(outer_grid[plot_idx])
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.ylim(-1,50)
ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
# ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax.set_yticks([0,10,20,30,40,50])
fig11.add_subplot(ax)

V_SMALL_SIZE = 18
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)

plot_idx = plot_idx + 1

for i in range(2000):
    
    start = time.time()
    
    i = i + 1
    
    # test_in = np.concatenate((potential,test_out)).reshape(1,-1)
    test_in = test_out.reshape(1,-1)
    test_out = model.predict(test_in)
    test_out_viz = test_out[0,:32]+test_out[0,32:]*1j
    
    test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
    test_out = test_out[0]
    test_out_viz = test_out[:32]+test_out[32:]*1j

    test_real = np.matmul(test_prop,test_real)

    overlap = get_den(np.sum(test_out_viz*np.conj(test_real)))
    
    pos_exp_real = np.sum(get_den(test_real)*xlist)
    pos_exp_pred = np.sum(get_den(test_out_viz)*xlist)
    
    pos_nrg_real = np.sqrt(get_den(np.matmul(np.matmul(ham,test_real),np.conj(test_real))))*627
    pos_nrg_pred = np.sqrt(get_den(np.matmul(np.matmul(ham,test_out_viz),np.conj(test_out_viz))))*627
    
    pos_real.append(pos_exp_real)
    nrg_real.append(pos_nrg_real)
    
    pos_pred.append(pos_exp_pred)
    nrg_pred.append(pos_nrg_pred)
    
    pos_error = pos_exp_pred-pos_exp_real
    nrg_error = pos_nrg_pred-pos_nrg_real

    print('\n')
    print('Step: ', i)
    print('Overlap: ', overlap)
    print('Position Expectation Error: ',pos_error)
    print('Energy Expectation Error: ',nrg_error)
    
    overlap_list.append(overlap)
    pos_error_list.append(pos_error)
    nrg_error_list.append(nrg_error)
    
    if i%400==0:
        
        spline_pot = interp1d(xlist, potential, kind='cubic')
        spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
        spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
        
        fig_pot = spline_pot(xlist_res)*627
        # fig_pot[np.where(fig_pot<0)] = 0
        
        fig_ann = spline_ann(xlist_res)*100
        # fig_ann = fig_ann - min(fig_ann)
        fig_ann[np.where(fig_ann<0)] = 0
        
        fig_real = spline_prp(xlist_res)*100
        # fig_real = fig_real - min(fig_real)
        fig_real[np.where(fig_real<0)] = 0
        
        V_SMALL_SIZE = 18
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 24
        
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        plt.ylim(-1,50)
        # plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)
        
        plot_exc = [1,2]
        plot_exc_2 = [4,5]    

        if plot_idx in plot_exc:
    
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        elif plot_idx in plot_exc_2:
            
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([])
            fig11.add_subplot(ax)
        
        else:  
        
            ax = fig11.add_subplot(outer_grid[plot_idx])
            # ax.ylim(-1,50)
            ax.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
            ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([0,10,20,30,40])
            fig11.add_subplot(ax)
            
        plot_idx = plot_idx + 1
        
        plt.text(-1.5, 2, str(i)+" fs", fontsize=SMALL_SIZE)

overlap_list = np.array(overlap_list)
pos_error_list = np.array(pos_error_list)
nrg_error_list = np.array(nrg_error_list)

pos_pred = np.array(pos_pred)
pos_real = np.array(pos_real)
nrg_pred = np.array(nrg_pred)
nrg_real = np.array(nrg_real)

plt.ylim(-1,50)
plt.show()

#%%

plt.plot(pos_pred/1.889,"r")
plt.plot(pos_real/1.889,"b--")
plt.show()

plt.plot(nrg_pred,"r")
plt.plot(nrg_real,"b--")
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(2, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_pred/1.889,"r")
ax.plot(pos_real/1.889,"b--")
ax.set_xticks([])
# ax.set_yticks([-0.25,0.0,0.25])
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_pred,"r")
ax.plot(nrg_real,"b--")
ax.set_xticks([0,400,800,1200,1600,2000])
# ax.set_yticks([5.0,6.0,7.0])
# plt.ylim(5.5,6.5)
fig11.add_subplot(ax)

plt.show()
#%%

plt.plot(overlap_list)
plt.show()

plt.plot(pos_error_list/1.889)
plt.show()

plt.plot(nrg_error_list)
plt.show()

#%%

fig11 = plt.figure(figsize=(6, 12), constrained_layout=False)
outer_grid = fig11.add_gridspec(3, 1, wspace=0.0, hspace=0.0)

i = 0
ax = fig11.add_subplot(outer_grid[i])
ax.plot(pos_error_list/1.889,'k')
ax.set_xticks([])
ax.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
plt.ylim(-0.1,0.1)
fig11.add_subplot(ax)
        
i = 1
ax = fig11.add_subplot(outer_grid[i])
ax.plot(nrg_error_list,'k')
ax.set_xticks([])
ax.set_yticks([-0.1,-0.05,0.0,0.05])
plt.ylim(-0.1,0.1)
fig11.add_subplot(ax)

i = 2
ax = fig11.add_subplot(outer_grid[i])
ax.plot(overlap_list,'k')
# ax.set_xticks([0,400,800,1200,1600,2000])
ax.set_yticks([0.9,0.95,1,1.05])
plt.ylim(0.9,1.1)
fig11.add_subplot(ax)


plt.show()

#%%

length = 10000
grid_size = 32

# test_chk = generate_pot_test(grid_size,0,0,1500,1500,10)[0]
test_chk = potential
test_chk = test_chk - min(test_chk)
test_pot = []
for i in range(length):
    test_pot.append(test_chk)
test_pot = np.array(test_pot)

xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 5

test_basis = generate_basis(xlist,test_pot[0],n_states)
test_state = generate_state(test_basis,n_states)
test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
test_out_viz = test_out[:32]+test_out[32:]*1j
test_out_real = test_out_viz

spline_pot = interp1d(xlist, test_pot[0], kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_out_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

plt.ylim(-1,50)
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.rc('font', family='Helvetica')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

i = 0
plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)      

plt.show()

test_real = test_out_viz

for i in range(1000):
    
    i = i + 1
    
    test_prop = generate_propagator(xlist,test_pot[i],1000)
    
    test_in = test_out.reshape(1,-1)
    test_out = model.predict(test_in)
    test_out_viz = test_out[0,:32]+test_out[0,32:]*1j

    test_real = np.matmul(test_prop,test_real)
    
    if i%100==0:
        
        print(i)
    
    if i%50==0:
        
        print(i)
        
        spline_pot = interp1d(xlist, test_pot[i], kind='cubic')
        spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
        spline_prp = interp1d(xlist, get_den(test_real), kind='cubic')
        
        fig_pot = spline_pot(xlist_res)*627
        # fig_pot[np.where(fig_pot<0)] = 0
        
        fig_ann = spline_ann(xlist_res)*100
        fig_ann[np.where(fig_ann<0)] = 0
        
        fig_real = spline_prp(xlist_res)*100
        fig_real[np.where(fig_real<0)] = 0
        
        plt.ylim(-1,50)
        plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_ann,"r",xlist_res/1.889,fig_real,"--b")
        
        plt.xlabel('Position (Å)')
        plt.ylabel('Energy (kcal/mol)')
        plt.rc('font', family='Helvetica')
        
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18
        
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)
    
        plt.show()

    test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
    test_out = test_out[0]
    
#%%

xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
potential = generate_pot_test_2(grid_size,0,1.8897*1.4,1500,1500,48)[0]
potential = potential - min(potential)

length = 350
grid_size = 32

test_pot = potential
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 2

ene = generate_spectra(xlist,potential,n_states)
test_basis = generate_basis(xlist,potential,n_states)
test_state = generate_state(test_basis,n_states)
# test_out = np.concatenate((np.real(test_state),np.imag(test_state)))

# test_out = np.exp(-((xlist+0.5*1.889)**2))
# test_out = test_out/np.sqrt(np.sum(get_den(test_out)))
# print(np.sum(test_out**2))
# test_out_real = test_out

test_out = (1/np.sqrt(0.5))*(test_basis[0] + test_basis[1])
test_out_real = test_out

plt.plot(get_den(test_out))
#%%

print(ene*627)

xlist_res = np.linspace(-1.5,1.5,1024)*1.8897
# test_out_viz = test_out[:32]+test_out[32:]*1j
# test_out_real = test_out_viz

spline_pot = interp1d(xlist, potential, kind='cubic')
spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
spline_prp = interp1d(xlist, get_den(test_out_real), kind='cubic')
spline_prp = interp1d(xlist, np.abs(test_out_real), kind='cubic')

fig_pot = spline_pot(xlist_res)*627
# fig_pot[np.where(fig_pot<0)] = 0

fig_ann = spline_ann(xlist_res)*100
fig_ann[np.where(fig_ann<0)] = 0

fig_real = spline_prp(xlist_res)*100
fig_real[np.where(fig_real<0)] = 0

plt.ylim(-0.5,4)
plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_real*0.025+ene[0]*627,"b")

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.rc('font', family='Helvetica')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

i = 0
# plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)      
plt.show()

#%%

test_real = test_out_viz

for i in range(25):
    
    i = i + 1
    
    test_prop = generate_propagator(xlist,potential,1000)
    test_in = np.concatenate((potential,test_out)).reshape(1,-1)
    test_out = model.predict(test_in)
    test_out_viz = test_out[0,:32]+test_out[0,32:]*1j
    
    test_out_viz = test_out_real

    test_real = np.matmul(test_prop,test_real)
    
    if i%1==0:
        
        print(i,get_den(np.sum(test_out_viz*np.conj(test_real))))
        
        spline_pot = interp1d(xlist, potential, kind='cubic')
        spline_ann = interp1d(xlist, get_den(test_out_viz), kind='cubic')
        spline_prp = interp1d(xlist, np.abs(test_real), kind='cubic')
        
        fig_pot = spline_pot(xlist_res)*627
        # fig_pot[np.where(fig_pot<0)] = 0
        
        fig_ann = spline_ann(xlist_res)*100
        fig_ann[np.where(fig_ann<0)] = 0
        
        fig_real = spline_prp(xlist_res)*100
        fig_real[np.where(fig_real<0)] = 0
        
        plt.ylim(-0.5,4)
        plt.plot(xlist_res/1.889,fig_pot,"k",xlist_res/1.889,fig_real*0.025+ene[0]*627,"b")
        plt.xlabel('Position (Å)')
        plt.ylabel('Energy (kcal/mol)')
        plt.rc('font', family='Helvetica')
        
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18
        
        plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # plt.text(-1.5, 5, "Time: "+str(i)+" fs", fontsize=SMALL_SIZE)
    
        plt.show()

    test_out = test_out/np.sqrt(np.sum(get_den(test_out_viz)))
    test_out = test_out[0]
    

#%%



