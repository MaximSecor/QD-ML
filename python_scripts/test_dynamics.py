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
    
    R_eq = (1.8*np.random.random())*1.8897
    X_eq = (40*np.random.random()-20)
    
    potential_1 = 0.5*K_p1*(xlist-(0.5*R_eq))**2
    potential_2 = 0.5*K_p2*(xlist+(0.5*R_eq))**2 + X_eq*(1/627)
    couplings = np.full((grid_size),10)*(1/627)
    
    two_state = np.array([[potential_1,couplings],[couplings,potential_2]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES_1 = ground_state_PES - np.min(ground_state_PES)
    
    return potential

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
    delta_t = dt/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

trials = 1000
time_steps = 1000
coeff_ave = np.zeros((10,time_steps))
state_en_ave = np.zeros((time_steps))

for t in range(trials):
    
    # fig = plt.figure(dpi=300)
    
    start = time.time()
    
    mass = 1836
    grid_size = 64
    xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
    test = generate_pot_TD(time_steps)
    
    # print(test[4],test[5])
    # for i in range(int(time_steps/10)):
    #     plt.ylim(-1,50)
    #     plt.plot(xlist/1.889,test[0][i*10]*627,'k',linewidth=0.1)
    # plt.plot(xlist/1.889,test[0][0]*627,':b',linewidth=2)
    # plt.plot(xlist/1.889,test[3]*627,':r',linewidth=2)
    # plt.show()
    
    n_states = 0
    traj = test[0]
    true_state_traj = []
    
    # ground_state_PES_1 = traj[0]
    # temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
    # basis_full = temp_full[1][:,0:n_states].T
    # state_coeff = 2*np.random.random(n_states)-1
    # state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
    # state_full = np.matmul(state_coeff,basis_full)
    # state_full = state_full[0]
    
    ground_state_PES_1 = traj[0]
    temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
    state_full = temp_full[1][:,n_states].T
    
    for t in range(time_steps):
        hamiltonian = prop_1D(xlist,traj[t],mass)
        delta_t = 1000/24.18
        prop_full = expm(-1j*hamiltonian*delta_t)
        state_full = np.matmul(state_full,prop_full)
        true_state_traj.append(state_full)
        
    true_state_traj = np.array(true_state_traj)
    
    end = time.time()
    print(end-start)
    
    # plt.plot(get_den(true_state_traj.T))
    # plt.show()
    
    # test_basis = []
    # for t in range(time_steps):
    #     temp_full = fgh_1D(xlist,traj[t],mass)
    #     # en_traj.append(temp_full[0][:10])
    #     basis_analysis = temp_full[1][:,:10].T
    #     test_basis.append(get_den(basis_analysis[0]))
    
    coeff_traj = []
    en_traj = []
    state_en_traj = []
    
    for t in range(time_steps):
        temp_full = fgh_1D(xlist,traj[t],mass)
        en_traj.append(temp_full[0][:10])
        basis_analysis = temp_full[1][:,:10].T
        coeff_traj.append((get_den(np.matmul(basis_analysis,true_state_traj[t]))))
        state_en_traj.append(np.sum((temp_full[0][:10])*(get_den(np.matmul(basis_analysis,true_state_traj[t])))))
    
    coeff_traj = np.array(coeff_traj)
    en_traj = np.array(en_traj)
    state_en_traj = np.array(state_en_traj)
    
    # print(coeff_traj.shape)
    
    # plt.plot(np.sum(coeff_traj[:,:1],1))
    # plt.plot(np.sum(coeff_traj[:,:2],1))
    # plt.plot(np.sum(coeff_traj[:,:5],1))
    # plt.plot(np.sum(coeff_traj[:,:10],1))
    # plt.show()
    
    for t in range(10):
        coeff_ave[t] += (np.sum(coeff_traj[:,:(t+1)],1))
    
    # plt.plot(en_traj*627)
    # plt.show()
    
    right = np.zeros((32)) + 1
    right = np.concatenate((right, np.zeros((32))))
    
    prot_side = []
    for t in range(time_steps):
        prot_side.append(np.sum(right*get_den(true_state_traj[t])))
    prot_side = np.array(prot_side)
    # plt.plot(prot_side)
    # plt.show()
    
    state_en_ave += state_en_traj
    
    # plt.plot(state_en_traj*627)
    # plt.show()
    
#%%

plt.ylim(-0.1,1.1)
plt.plot(coeff_ave[4]/trials)
plt.show()

plt.ylim(0,10)
plt.plot((state_en_ave/trials)*627)
plt.show()

#%%

for i in range(10):
    print(coeff_ave[i][999]/trials)

#%%

for i in range(100):
    print((state_en_ave[i*10]/trials)*627)

#%%

fig = plt.figure(dpi=100)
camera = Camera(fig)

for i in range(int(time_steps/10)):
    
    plt.text(0.75, 45, "Time Elapsed: "+str(str(int(i))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))
    
    plt.ylim(-1,50)
    # plt.plot(xlist/1.889,test[0][i]*627,'k')
    plt.plot(xlist,get_den(true_state_traj[i])*500,'b',xlist,traj[i]*627,'k')
    # plt.plot(xlist/1.889,test_basis[i]*250,'k')
    
    plt.xlabel('x (bohr)')
    plt.ylabel('Energy (kcal/mol)')
    
    plt.rc('font', family='Helvetica')
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    camera.snap()
    
animation = camera.animate(interval=20)
animation.save('/Users/maximsecor/Desktop/potentials.gif', writer = 'imagemagick')

