#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:20:27 2021

@author: maximsecor
"""

#%%

import numpy as np
from numba import jit
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import math 

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=4,suppress=True)

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

def generate_basis(ground_state_PES_1,n_states):
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    temp = fgh_1D(xlist,ground_state_PES_1,mass)
    basis = temp[1][:,0:n_states].T
    return basis

def generate_state(basis,n_states):
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state = np.matmul(state_coeff,basis)
    state = state[0]
    
    return state

def generate_propagator_nth(test_pot_2,time_step,n_order):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
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
        
        # print((1/(math.factorial(i))),((-1j)**i))
        
        prop_s = prop_s + prop_add
            
    return prop_s

def generate_propagator_exact(test_pot_2,time_step):

    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = time_step/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

start = time.time()            

trials = 2
orders = np.linspace(1,128,128,dtype=int)
truncation_results = np.zeros((len(orders)))

for q in range(trials):         
    
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

    traj = []
    
    for t in range(10):
        
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
        
        traj.append(ground_state_PES_1)
    
    traj = np.array(traj)
    
    test_basis = generate_basis(ground_state_PES_1,n_states)
    test_state = generate_state(test_basis,n_states)
    
    tracker_data = np.zeros((len(orders)))
    
    for i in range(len(orders)):
        
        test_state_10 = test_state
        test_state_exact = test_state
        
        for t in range(10):
    
            test_prop_10 = (generate_propagator_nth(traj[t],1000,orders[i]))
            test_prop_exact = generate_propagator_exact(traj[t],1000)
            
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

