#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:07:00 2021

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
    
    grid = 32
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
    minimum_trajectory = []
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
        minimum_trajectory.append(np.min(ground_state_PES))
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
    
    return potential_trajectory,minimum_trajectory,test_pos,test_solv,ground_state_PES_1,R_eq,R_0

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

length = 1000
test = generate_pot_TD(length)

grid = 32
n_states = 1
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897

plt.ylim(-1,50)
plt.plot(xlist,test[0][0]*627)

test_basis = generate_basis(xlist,test[0][0],n_states)
test_state_start = generate_state(test_basis,n_states)

plt.plot(xlist,get_den(test_state_start)*100)

#%%

test_state_1 = test_state_start
for i in range(length):
    test_prop_exact = generate_propagator_exact(xlist,test[0][i],1000)
    test_state_1 = np.matmul(test_state_1,test_prop_exact)
    
plt.ylim(-1,50)
plt.plot(xlist,test[0][i]*627)
plt.plot(xlist,get_den(test_state_1)*100)

test_state_2 = test_state_start
for i in range(length):
    test_prop_exact = generate_propagator_exact(xlist,test[0][i]+test[1][i],1000)
    test_state_2 = np.matmul(test_state_2,test_prop_exact)
    
plt.ylim(-1,50)
plt.plot(xlist,test[0][i]*627)
plt.plot(xlist,get_den(test_state_2)*100)

delta_t = 1000/24.18
transfer_phase = 1
for i in range(length):
    transfer_phase = transfer_phase*np.exp(-1j*test[1][i]*delta_t)
print(transfer_phase)

test_state_3 = test_state_1*transfer_phase

test_result = np.sum(test_state_1*np.conj(test_state_1))
print(test_result)

test_result = np.sum(test_state_2*np.conj(test_state_2))
print(test_result)

test_result = np.sum(test_state_1*np.conj(test_state_2))
print(test_result)

test_result = np.sum(test_state_3*np.conj(test_state_2))
print(test_result)

test_result = np.sum(test_state_1*np.conj(test_state_3))
print(test_result)
    
#%%

start = time.time()    

test_state_1 = test_state_start
for i in range(length):
    test_prop_exact = generate_propagator_exact(xlist,test[0][i],1000)
    test_state_1 = np.matmul(test_state_1,test_prop_exact)
    
end = time.time() 
print(end-start) 

#%%

length = 1000
test = generate_pot_TD(length)

grid = 32
n_states = 1
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897

plt.ylim(-1,50)
plt.plot(xlist,test[0][0]*627)

test_basis = generate_basis(xlist,test[0][0],n_states)
test_state_start = generate_state(test_basis,n_states)

plt.plot(xlist,get_den(test_state_start)*100)
plt.show()
    
start = time.time()    

length = 1000
delta_t = 1000/24.18
test_state_1 = test_state_start
for i in range(length):
    temp_soln = fgh_1D(xlist,test[0][i],1836)
    # print(temp_soln[0][:5])
    state_coeff = np.matmul(test_state_1,temp_soln[1][:,:10])
    # print(get_den(state_coeff))
    test_prop_eigen = np.exp(-1j*temp_soln[0][:10]*delta_t)
    state_coeff = state_coeff*test_prop_eigen
    test_state_1 = np.matmul(temp_soln[1][:,:10],state_coeff)
    
# state_coeff = np.matmul(test_state_1,temp_soln[1][:,:5])
# print(get_den(state_coeff))
    
final_coeff_test = test_state_1
 
end = time.time() 
print(end-start) 
    
start = time.time()    

length = 1000
delta_t = 1000/24.18
test_state_1 = test_state_start
for i in range(length):
    test_prop_exact = generate_propagator_exact(xlist,test[0][i],1000)
    test_state_1 = np.matmul(test_state_1,test_prop_exact)
    temp_soln = fgh_1D(xlist,test[0][i],1836)
    state_coeff = np.matmul(test_state_1,temp_soln[1][:,:5])

print(get_den(state_coeff))
    
end = time.time() 
print(end-start) 

test_result = (np.sum(final_coeff_test*np.conj(test_state_1)))
print(test_result)

plt.ylim(-1,50)
plt.plot(xlist,test[0][0]*627)
plt.plot(xlist,get_den(final_coeff_test)*100)
plt.plot(xlist,get_den(test_state_1)*100)
plt.show()








    