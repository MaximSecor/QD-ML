#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:48:04 2020

@author: maximsecor
"""

#%%

import numpy as np
from numba import jit
import time
import os
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(precision=4,suppress=True)

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

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

def generate_basis(ground_state_PES_1):
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    temp = fgh_2d(xlist,ground_state_PES_1,mass)
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
    hamiltonian = pot_2d(xlist,test_pot_2,mass)
    delta_t = 1000/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

start = time.time()

non_stationaty_states_present = []
non_stationaty_states_future = []
acting_hamiltonian = []

n_states = 5
mass = 1836
grid_size = 32

test_pot_2 = generate_pot()
prop = generate_propagator(test_pot_2)

for k in range(1):
    
    print(k)
    
    test_pot_1 = generate_pot()
    basis = generate_basis(test_pot_1)
    
    for i in range(100):
        
        state = generate_state(basis)

        for j in range(100):
            
            acting_hamiltonian.append(test_pot_2.reshape(1024))
        
            phase = (2*np.random.random()-1)*np.pi
            state_RP = state * np.exp(1j*phase)
            
            non_stationaty_states_present.append(state_RP)
            state_future = np.matmul(prop,state_RP)
            non_stationaty_states_future.append(state_future)
            state = state_future
            
non_stationaty_states_present = np.array(non_stationaty_states_present)
non_stationaty_states_future = np.array(non_stationaty_states_future)
acting_hamiltonian = np.array(acting_hamiltonian)

end = time.time()
print(end-start)

present_real = np.real(non_stationaty_states_present)
present_imag = np.imag(non_stationaty_states_present)
present = np.concatenate((present_real,present_imag),1)
# present = np.concatenate((acting_hamiltonian,present),1)

print(present.shape)

future_real = np.real(non_stationaty_states_future)
future_imag = np.imag(non_stationaty_states_future)
future = np.concatenate((future_real,future_imag),1)

print(future.shape)

file_present = '/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/file_present_1.csv'
file_future = '/Users/maximsecor/Desktop/TDSE_FRESH/TDSE_EXAMPLES/TIH_2D/file_future_1.csv'

os.system('touch ' + file_present)
os.system('touch ' + file_future)

df_features_potential = pd.DataFrame(present)
df_target_energies = pd.DataFrame(future)

df_features_potential.to_csv(file_present, index = False, header=True)
df_target_energies.to_csv(file_future, index = False, header=True)



