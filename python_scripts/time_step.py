#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:18:52 2021

@author: maximsecor
"""

import numpy as np
from numba import jit
import time
import os
import pandas as pd

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

def generate_pot_TD(grid_size,length,dt):

    dt_fs = 1000/24.188
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
        
        R_coord = (R_0-R_eq)*(np.cos(R_freq*t*dt_fs*dt+phase_1))+R_eq
        X_coord = (lam)*(np.cos(X_freq*t*dt+phase_2))+X_eq
        
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
    
    R_eq = (1.8*np.random.random())*1.8897
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

def generate_propagator(xlist,potential,dt):
    
    hamiltonian = prop_1D(xlist,potential,1836)
    delta_t = (1000*dt)/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%
    
fig = plt.figure(dpi=600)

grid_size = 128
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
test = generate_pot_TD(grid_size,100000,0.001)
equil = test[3]
start = test[0][0]
print(test[0].shape)

plt.ylim(-1,50)
for i in range(50):
    plt.plot(test[0][i*2000]*627,'k',linewidth=0.1)
    
plt.plot(equil*627,':r',linewidth=2)
plt.plot(start*627,':b',linewidth=2)

#%%
    
trials = 10
length = 900

start_time_factor = 10
time_factor = [3,10,30,100,300,1000,3000]

error = np.zeros((len(time_factor)))

grid_size = 32
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 1

start = time.time()

for q in range(trials):
    
    print(q)
    
    traj_full = generate_pot_TD(grid_size,int((length*1000)/start_time_factor),0.001*start_time_factor)
    traj_potential = traj_full[0]
    
    ground_state_PES_1 = traj_potential[0]
    temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
    basis_full = temp_full[1][:,0:n_states].T
    
    state_coeff = 2*np.random.random(n_states)-1
    state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    state_full = np.matmul(state_coeff,basis_full)
    state_start = state_full[0]
    state_full = state_start
    
    true_state_traj = []
    for t in range(len(traj_potential)):
        prop = generate_propagator(xlist,traj_potential[t],0.001*start_time_factor) 
        state_full = np.matmul(state_full,prop)
        true_state_traj.append(state_full)
    true_state_traj = np.array(true_state_traj)
    
    for f in range(len(time_factor)):
        
        state_test = state_start
        for t in range(int(len(traj_potential)/time_factor[f])):
            prop = generate_propagator(xlist,traj_potential[int(t*time_factor[f])],0.001*start_time_factor*time_factor[f]) 
            state_test = np.matmul(state_test,prop)
        error[f] += (get_den(np.sum(state_test*np.conj(true_state_traj[-1]))))

print(error/trials)
    
end = time.time()
print(end-start)

#%%

test = np.linspace(0,100,101)
print(test[-1])

#%%

plt.ylim(-1,50)


plt.plot(traj_potential[0]*627)
plt.plot(get_den(true_state_traj)[0]*100)

#%%

fig = plt.figure(dpi=300)
camera = Camera(fig)

for i in range(1000):
    
    tstep = i
    
    plt.text(0.75, 45, "Time Elapsed: "+str(str(int(tstep))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))

    plt.ylim(-5,50)
    plt.plot(xlist,get_den(true_state_traj[tstep])*100,'b',xlist,traj_potential[tstep]*627,'k')
    
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
    
animation = camera.animate(interval=20)
animation.save('/Users/maximsecor/Desktop/potentials.gif', writer = 'imagemagick')

#%%

# factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000])
factors = np.array([10,100,1000,10000])
results = np.zeros((len(factors)))

for q in range(10):

    start = time.time()
    
    grid_size = 32
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
    dt_da = 0.001/0.024188
    
    f0 = 55.734
    mu = 0.2652
    lam = (5*np.random.random())
    x_solv = np.sqrt(lam/(2*f0))
    freq_solv = np.sqrt(f0/mu)
    dt_solv = 0.000001
    
    traj = []
    
    for t in range(10000):
        
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
    
    # start = time.time()
    
    final_states = []
    
    n_states = 5
    test_basis = generate_basis(ground_state_PES_1,n_states)
    test_state = generate_state(test_basis,n_states)
    
    # factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000,25000,50000])
    # factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000])
    
    for f in range(len(factors)):
    
        test_state_exact = test_state
        
        for t in range(int(10000/factors[f])):
            
            test_prop_exact = generate_propagator_exact(traj[t*factors[f]],factors[f])
            test_state_exact = np.matmul(test_state_exact,test_prop_exact)
    
        final_states.append(test_state_exact)
    
    end = time.time()
    
    # print(end-start)
    
    final_states = np.array(final_states)
    
    for i in range(len(factors)):
        results[i] = results[i] + (np.sqrt(get_den(np.sum(final_states[0]*np.conj(final_states[i])))))

for i in range(len(factors)):
    print(results[i]/10)