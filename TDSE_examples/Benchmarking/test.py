#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:56:39 2021

@author: maximsecor
"""

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

def generate_pot_TD(length,coup_const):

    grid = 512
    mass = 1836
    
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    KX_1 = (2500*np.random.random()+1500)/(350*627)
    KX_2 = (2500*np.random.random()+1500)/(350*627)
    dE = (20*np.random.random()-10)/627
    
    red_mass = ((12*12)/(12+12))*1836
    coord_eq = 1.25*np.random.random()*1.8897
    x_da = 1.25*np.random.random()*1.8897
    v_da = 0
    a_da = 0
    freq_da = 200*np.random.random()+100
    k_const = red_mass*(freq_da/(627*350))**2
    dt_da = 1/0.024188
    
    f0 = 55.734
    mu = 0.2652
    lam = (10*np.random.random())
    x_solv = np.sqrt(lam/(2*f0))
    freq_solv = np.sqrt(f0/mu)
    dt_solv = 0.001

    potential_trajectory = []
    test_pos = []
    test_solv = []
    
    for t in range(length):
        
        if x_da < 0:
            v_da = np.abs(v_da)
        
        x_da = x_da + v_da*dt_da + 0.5*a_da*dt_da**2
        a_da_2 = a_da
        a_da = -k_const*(x_da-coord_eq)/red_mass
        v_da = v_da + 0.5*(a_da_2+a_da)*dt_da
         
        x_solv_t = x_solv*np.cos(freq_solv*t*dt_solv)
        solv_gap = x_solv_t*(np.sqrt(2*f0*lam))
        
        potential_1 = 0.5*mass*(KX_1**2)*(xlist-(0.5*x_da))**2
        potential_2 = 0.5*mass*(KX_2**2)*(xlist+(0.5*x_da))**2 + dE + solv_gap*(1/627)
        couplings = np.full((grid_size),coup_const)*(1/627)
        
        two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
        ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
        ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
        
        potential_trajectory.append(ground_state_PES_1)
        test_pos.append(x_da)
        test_solv.append(dE+solv_gap*(1/627))
        
    potential_trajectory = np.array(potential_trajectory)
    test_pos = np.array(test_pos)
    test_solv = np.array(test_solv)
    
    potential_1 = 0.5*mass*(KX_1**2)*(xlist-(0.5*coord_eq))**2
    potential_2 = 0.5*mass*(KX_2**2)*(xlist+(0.5*coord_eq))**2 + dE
    couplings = np.full((grid_size),coup_const)*(1/627)
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return potential_trajectory,test_pos,test_solv,ground_state_PES_1

def generate_pot(grid_size):
    
    mass = 1836
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    KX_1 = (2500*np.random.random()+1500)/(350*627)
    KX_2 = (2500*np.random.random()+1500)/(350*627)
    dE = (40*np.random.random()-20)/627
    x_da = 2.5*np.random.random()*1.8897
    
    diabatic_1 = 0.5*mass*(KX_1**2)*(xlist-(0.5*x_da))**2
    diabatic_2 = 0.5*mass*(KX_2**2)*(xlist+(0.5*x_da))**2 + dE
    couplings = np.full((grid_size),10)*(1/627)
    
    two_state = np.array([[diabatic_1,couplings],[couplings,diabatic_2]])
    potential_gs = np.linalg.eigh(two_state.T)[0].T[0]
    potential = potential_gs - np.min(potential_gs)
    
    return potential

def generate_pot_test(grid_size,dE,x_da,KX_1,KX_2,couplings):
    
    mass = 1836
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    KX_1 = (KX_1)/(350*627)
    KX_2 = (KX_2)/(350*627)
    dE = dE/627
    x_da = x_da*1.8897
    
    diabatic_1 = 0.5*mass*(KX_1**2)*(xlist-(0.5*x_da))**2 - 0.5*dE
    diabatic_2 = 0.5*mass*(KX_2**2)*(xlist+(0.5*x_da))**2 + 0.5*dE
    couplings = np.full((grid_size),couplings)*(1/627)
    
    two_state = np.array([[diabatic_1,couplings],[couplings,diabatic_2]])
    
    potential_gs = np.linalg.eigh(two_state.T)[0].T[0]
    # potential_gs = potential_gs - np.min(potential_gs)
    
    potential_es = np.linalg.eigh(two_state.T)[0].T[1]
    # potential_es = potential_es - np.min(potential_es)
    
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

prot_surf_1 = []
prot_surf_2 = []    

for i in range(40):
    
    grid_size = 128
    test = generate_pot_test(grid_size,20-i,1.0,3000,3000,10)
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    plt.ylim(-50,250)
    plt.plot(xlist/1.8897,test[0]*627)
    ham_sol = fgh_1D(xlist,test[0],1836)
    prot_surf_1.append(ham_sol[0][:5])
    
    plt.ylim(-50,250)
    plt.plot(xlist/1.8897,test[1]*627)
    ham_sol = fgh_1D(xlist,test[1],1836)
    prot_surf_2.append(ham_sol[0][:5])
    
    # plt.plot(xlist/1.8897,get_den(ham_sol[1][:,0])*200)

prot_surf_1 = np.array(prot_surf_1)*627   
prot_surf_2 = np.array(prot_surf_2)*627   

#%%

energy_list = np.linspace(-1.5,1.5,40)*1.8897
for i in range(5):
    plt.plot(prot_surf_1[:,i]+5*energy_list**2)
plt.show()

for i in range(5):
    plt.plot(prot_surf_2[:,i])
plt.show()

#%%

fig = plt.figure(dpi=300)
for q in range(1):

    grid_size = 512
    test = generate_pot_TD(1000,10)
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    
    for i in range(100):
        plt.ylim(-10,50)
        plt.plot(xlist/1.889,test[0][i*10]*627,'k',linewidth=0.1)
        
    plt.plot(xlist/1.889,test[0][0]*627,':b',linewidth=2)
    plt.plot(xlist/1.889,test[3]*627,':r',linewidth=2)
    plt.show()   
    # plt.savefig('/Users/maximsecor/Desktop/extra.png')
    
#%%
    
print(test.shape)