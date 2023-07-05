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

for f in range(7):
    
    f = f

    for t in range(1):
        
        trials = 500
        trajectories = 0 + t*100
        
        position_agg_err = 0
        overlap_agg_err = 0
        density_agg_err = 0
             
        for q in range(trials):
            
            print(q)
            
            grid_size = 1024
            mass = 1836
            factor = 2**f
            
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
            
            n_states = 5
            
            temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
            basis_full = temp_full[1][:,0:n_states].T
            
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k] 
            
            state_coeff = 2*np.random.random(n_states)-1
            state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
            
            state_full = np.matmul(state_coeff,basis_full)
            state_full = state_full[0]
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            # plt.ylim(-1,50)
            # plt.plot(xlist,ground_state_PES_1*627)
            # plt.plot(xlist,get_den(state_full)*250)
            # plt.plot(xlist[::factor],get_den(state_sparse)*250)
            
            hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
            delta_t = 1000/24.18
            prop_full = expm(-1j*hamiltonian*delta_t)
            
            hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            delta_t = 1000/24.18
            prop_sparse = expm(-1j*hamiltonian*delta_t)
            
            err = 0
            
            for i in range(trajectories):
                
                state_sparse = np.matmul(state_sparse,prop_sparse)
                state_full = np.matmul(state_full,prop_full)
                
                # plt.ylim(-1,50)
                # plt.plot(xlist,ground_state_PES_1*627)
                # plt.plot(xlist,get_den(state_full)*250*factor)
                # plt.plot(xlist[::factor],get_den(state_sparse)*250)
                # plt.show()
                
                # err += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
                # err += np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor))
                # err += get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor)))
            
            # print(err/trajectories)
            
            position_agg_err += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
            overlap_agg_err += (get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            density_agg_err += np.sum(np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor)))
            # print(get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            
        print("trial error",f,t,overlap_agg_err/trials,position_agg_err/trials,density_agg_err/trials)

# plt.ylim(-1,50)
# plt.plot(xlist,ground_state_PES_1*627)
# plt.plot(xlist,get_den(state_full)*250*factor)
# plt.plot(xlist[::factor],get_den(state_sparse)*250)
# plt.show()
        
#%%

start = time.time()

position_agg_err = np.zeros((6,10))
overlap_agg_err = np.zeros((6,10))
density_agg_err = np.zeros((6,10))

trials = 100
     
for q in range(trials):
    
    print(q)
    
    grid_size = 1024
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
    
    for f in range(5):
        
        factor = 2**(f+1)
        
        for i in range(10):
            
            n_states = i + 1
            
            basis_full = temp_full[1][:,0:n_states].T
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k] 
            
            state_coeff = 2*np.random.random(n_states)-1
            state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
            
            state_full = np.matmul(state_coeff,basis_full)
            state_full = state_full[0]
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            # hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
            # delta_t = 1000/24.18
            # prop_full = expm(-1j*hamiltonian*delta_t)
            
            # hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            # delta_t = 1000/24.18
            # prop_sparse = expm(-1j*hamiltonian*delta_t)
            
            position_agg_err[f,i] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
            overlap_agg_err[f,i] += (get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            density_agg_err[f,i] += np.sum(np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor)))
        
print("trial error",'\n',overlap_agg_err/trials,'\n',position_agg_err/trials,'\n',density_agg_err/trials)

end = time.time()

print(end-start)

print(temp_sparse[0][:5])
print(temp_full[0][:5])
        
plt.ylim(-1,50)
plt.plot(xlist,ground_state_PES_1*627)
plt.plot(xlist,get_den(state_full)*100*factor)
plt.plot(xlist[::factor],get_den(state_sparse)*100)
plt.show()
     
#%%

for i in range((overlap_agg_err).shape[0]):
    print(int(1024/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(density_agg_err[i,f]/trials)

#%%

print(temp_sparse[0][:5])
print(temp_full[0][:5])
        
plt.ylim(-1,50)
plt.plot(xlist,ground_state_PES_1*627)
plt.plot(xlist,get_den(state_full)*100*factor)
plt.plot(xlist[::factor],get_den(state_sparse)*100)
plt.show()
        
print(state_sparse.shape)

#%%

for k in range(10):
    
    if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
        temp_sparse[1][:,k] = -1*temp_sparse[1][:,k] 
    
    plt.ylim(-50,50)
    plt.plot(xlist,ground_state_PES_1*627)
    plt.plot(xlist,temp_full[1][:,k]*50*np.sqrt(factor))
    plt.plot(xlist[::factor],temp_sparse[1][:,k]*50)
    plt.show()
    
#%%
    
start = time.time()

position_agg_err = np.zeros((6,10))
overlap_agg_err = np.zeros((6,10))
density_agg_err = np.zeros((6,10))

trials = 1
time_steps = 0
     
for q in range(trials):
    
    print(q)
    
    grid_size = 1024
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
    
    for f in range(1):
        
        factor = 2**(f+4)
        print(1024/factor)
        
        for i in range(10):
            
            n_states = i + 1
            
            basis_full = temp_full[1][:,0:n_states].T
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k] 
            
            state_coeff = 2*np.random.random(n_states)-1
            state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
            
            state_full = np.matmul(state_coeff,basis_full)
            state_full = state_full[0]
            
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
            delta_t = 1000/24.18
            prop_full = expm(-1j*hamiltonian*delta_t)
            
            hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            delta_t = 1000/24.18
            prop_sparse = expm(-1j*hamiltonian*delta_t)
            
            for t in range(time_steps):
                state_full = np.matmul(state_full,prop_full)
                state_sparse = np.matmul(state_sparse,prop_sparse)
                
            plt.ylim(-1,50)
            plt.title(str(int(1024/factor))+" "+str(int(i)))
            plt.plot(xlist,ground_state_PES_1*627)
            plt.plot(xlist,get_den(state_full)*100*factor)
            plt.plot(xlist[::factor],get_den(state_sparse)*100)
            plt.show()
            
            position_agg_err[f,i] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
            overlap_agg_err[f,i] += (get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            density_agg_err[f,i] += np.sum(np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor)))
        
# print("trial error",'\n',overlap_agg_err/trials,'\n',position_agg_err/trials,'\n',density_agg_err/trials)

end = time.time()

print(end-start)

print(temp_sparse[0][:5])
print(temp_full[0][:5])
        
plt.ylim(-1,50)
plt.plot(xlist,ground_state_PES_1*627)
plt.plot(xlist,get_den(state_full)*100*factor)
plt.plot(xlist[::factor],get_den(state_sparse)*100)
plt.show()

#%%

start = time.time()

test_states = 5
trials = 5
time_steps = 1000

position_agg_err = np.zeros((3,test_states))
overlap_agg_err = np.zeros((3,test_states))
density_agg_err = np.zeros((3,test_states))

for i in range(test_states):
    for q in range(trials):
        
        print(i,q)
        
        n_states = i + 1
        
        grid_size = 1024
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
        
        hamiltonian = prop_1D(xlist,ground_state_PES_1,mass)
        delta_t = 1000/24.18
        prop_full = expm(-1j*hamiltonian*delta_t)
        
        for t in range(time_steps):
            state_full = np.matmul(state_full,prop_full)
        
        for f in range(3):
            
            factor = 2**(f+4)
            # print(1024/factor)
            
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k]
                    
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            hamiltonian = prop_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            delta_t = 1000/24.18
            prop_sparse = expm(-1j*hamiltonian*delta_t)
            
            for t in range(time_steps):
                state_sparse = np.matmul(state_sparse,prop_sparse)
                
            plt.ylim(-1,50)
            plt.title(str(int(1024/factor))+" "+str(int(i)))
            plt.plot(xlist,ground_state_PES_1*627)
            plt.plot(xlist,get_den(state_full)*100*factor)
            plt.plot(xlist[::factor],get_den(state_sparse)*100)
            plt.show()
            
            position_agg_err[f,i] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(state_full)*xlist)))
            overlap_agg_err[f,i] += (get_den(np.sum(state_sparse*np.conj(state_full[::factor])*np.sqrt(factor))))
            density_agg_err[f,i] += np.sum(np.abs((get_den(state_sparse)-get_den(state_full[::factor])*factor)))               
                
end = time.time()

print(end-start)

#%%

print(get_den(state_coeff))
print(get_den(np.matmul(state_full.T,basis_full.T)))
print(get_den(np.matmul(state_sparse.T,basis_sparse.T)))

#%%

print("\n OVERLAP ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(1024/(2**(i+4))))
    for f in range((overlap_agg_err).shape[1]):
        print(overlap_agg_err[i,f]/trials)
      
print("\n DENSITY ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(1024/(2**(i+4))))
    for f in range((overlap_agg_err).shape[1]):
        print(density_agg_err[i,f]/trials)

print("\n POSITION ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(1024/(2**(i+4))))
    for f in range((overlap_agg_err).shape[1]):
        print(position_agg_err[i,f]/trials)
        
        