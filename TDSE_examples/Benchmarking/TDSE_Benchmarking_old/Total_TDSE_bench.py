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

def generate_propagator(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 0.01/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

def generate_propagator_simple(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 0.01/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t
    
    return prop_s

#%%
    
grid_size = 128
mass = 1836

test_pot = generate_pot()
plt.ylim(-1,50)
plt.plot(test_pot*627)
plt.show()

n_states = 5

#%%

test_basis = generate_basis(test_pot,n_states)
test_state = generate_state(test_basis,n_states)
test_state_exact = test_state
test_prop = generate_propagator(test_pot)
test_prop_simple = generate_propagator_simple(test_pot)

#%%

print(np.max(get_den(test_prop)))
print(np.max(get_den(test_prop_simple)))

#%%

for j in range(10):
    for i in range(1000000):
        
        test_state = np.matmul(test_state,test_prop_simple)
        # test_state = test_state / np.sqrt(np.sum(get_den(test_state)))
        test_state_exact = np.matmul(test_state_exact,test_prop)
        
    plt.ylim(-1,50)
    plt.plot(test_pot*627)
    plt.plot(get_den(test_state)*100+5)
    plt.plot(get_den(test_state_exact)*100)
    plt.show()
    
#%%
    
    
    




