#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:07:39 2021

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

def generate_propagator_exact(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 1/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

def generate_propagator_first(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 1/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t
    
    return prop_s

def generate_propagator_second(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = 1/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t-0.5*np.matmul(hamiltonian,hamiltonian)*delta_t**2
    
    return prop_s

#%%

start = time.time()

test_states = 1
trials = 10
time_steps = 1
grid_starter = 1024
grid_factors = 7

position_agg_err = np.zeros((grid_factors,test_states))
overlap_agg_err = np.zeros((grid_factors,test_states))
density_agg_err = np.zeros((grid_factors,test_states))

for i in range(test_states):
    for q in range(trials):
        
        print(i,q)
        
        n_states = i + 1
        
        grid_size = grid_starter
        mass = 1836
        
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
        
        temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
        basis_full = temp_full[1][:,0:n_states].T
        
        state_coeff = 2*np.random.random(n_states)-1
        state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
        
        state_full = np.matmul(state_coeff,basis_full)
        state_full = state_full[0]
        
        # hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
        # delta_t = 1000/24.18
        # prop_full = expm(-1j*hamiltonian*delta_t)
        
        for t in range(time_steps):
            hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
            delta_t = 1000/24.18
            prop_full = expm(-1j*hamiltonian*delta_t)
            state_full = np.matmul(state_full,prop_full)
        
        for f in range(grid_factors):
            
            factor = 2**(f+1)
            print(grid_starter/factor)
            
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k]
                    
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            # hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            # delta_t = 1000/24.18
            # prop_sparse = expm(-1j*hamiltonian*delta_t)
            
            for t in range(time_steps):
                hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
                delta_t = 1000/24.18
                prop_sparse = expm(-1j*hamiltonian*delta_t)
                state_sparse = np.matmul(state_sparse,prop_sparse)
            
            position_agg_err[f,i] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
            overlap_agg_err[f,i] += (get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            density_agg_err[f,i] += np.sum(np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor)))               
                
end = time.time()

print(end-start)

print("\n OVERLAP ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(overlap_agg_err[i,f]/trials)
      
print("\n DENSITY ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(density_agg_err[i,f]/trials)

print("\n POSITION ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(position_agg_err[i,f]/trials)
        
        