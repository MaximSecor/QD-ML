#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:20:27 2021

@author: maximsecor
"""

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=10,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

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
    
    grid = 64
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
    # ground_state_PES_1[ground_state_PES_1>0.159489] = 0.159489
    ground_state_PES_1[ground_state_PES_1>0.07974] = 0.07974
    
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

def generate_propagator_nth(xlist,potential,time_step,n_order):
    
    mass = 1836
    hamiltonian = prop_1D(xlist,potential,mass)
    delta_t = time_step/24.18
    prop_s = np.eye(len(hamiltonian))
    
    for i in range(n_order):
        
        i = i+1
        
        prop_add = 1
        hamiltonian_add = hamiltonian
        for k in range(i-1):
            hamiltonian_add = np.matmul(hamiltonian,hamiltonian_add)
        prop_add = prop_add*hamiltonian_add
        prop_add = prop_add*(delta_t**i)
        prop_add = prop_add*((-1j)**i)
        prop_add = prop_add*(1/(math.factorial(i)))
        
        prop_s = prop_s + prop_add
            
    return prop_s

def generate_propagator_exact(xlist,test_pot_2,time_step):

    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = time_step/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%
   
print(50/627)    

#%%
    
n_states = 2
grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
ground_state_PES_1 = generate_pot(grid_size)
test_basis = generate_basis(xlist,ground_state_PES_1,n_states)
test_state = generate_state(test_basis,n_states)
   
plt.ylim(-1,50)
plt.plot(xlist/1.8897,ground_state_PES_1*627)
plt.plot(xlist/1.8897,get_den(test_state)*100)

#%%

start = time.time()

trials = 100
n_states = 5
orders = np.linspace(1,128,128,dtype=int)
truncation_results = np.zeros((len(orders)))

for q in range(trials):         
    
    print(q)
    grid_size = 32
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    mass = 1836
    
    ground_state_PES_1 = generate_pot(grid_size)
    test_basis = generate_basis(xlist,ground_state_PES_1,n_states)
    test_state = generate_state(test_basis,n_states)
    test_prop_exact = generate_propagator_exact(xlist,ground_state_PES_1,1000)
    
    tracker_data = np.zeros((len(orders)))
    
    for i in range(len(orders)):
        
        test_state_10 = test_state
        test_state_exact = test_state
        test_prop_10 = generate_propagator_nth(xlist,ground_state_PES_1,1000,orders[i])
        
        for t in range(1000):
            
            test_state_10 = np.matmul(test_state_10,test_prop_10)
            test_state_exact = np.matmul(test_state_exact,test_prop_exact)
        
        tracker_data[i] = get_den(np.sum(test_state_10*np.conj(test_state_exact)))
        
        if str(tracker_data[i]) != str(math.nan):
            checker = np.abs(tracker_data[i] - 1)
            if checker < 0.01:
                truncation_results[i] = truncation_results[i]+1
                
end = time.time() 
print(end-start)

for i in range(len(orders)):
    print(truncation_results[i]/trials)


#%%

plt.ylim(-1,50)
plt.plot(xlist/1.8897,ground_state_PES_1*627)
plt.plot(xlist/1.8897,get_den(test_state_exact)*100)

#%%

# orders = np.linspace(1,128,128,dtype=int)
# time_test_exact = np.zeros((len(orders)))

# trials = 100

# for q in range(trials):
#     start = time.time()
#     test_prop_exact = generate_propagator_exact(traj[t],1000)
#     end = time.time()
#     time_test_exact += (end-start) 

# time_test_exact /= trials

# time_test = np.zeros((len(orders)))

# for q in range(trials):
#     print(q)
#     for i in range(len(orders)):
        
#         start = time.time()
#         test_prop_10 = (generate_propagator_nth(traj[t],1000,orders[i]))
#         end = time.time()
#         time_test[i] += (end-start)
    
# plt.plot(time_test_exact)
# plt.plot(time_test/trials)

# #%%


# print(time_test_exact[0])
# print(time_test[13]/trials)

