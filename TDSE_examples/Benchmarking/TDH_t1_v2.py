#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:48:40 2021

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
    delta_t = dt/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

#%%

grid_size = 512
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897

plt.ylim(-1,50)
plt.plot(xlist/1.8897,generate_pot(grid_size)*627)

#%%

start = time.time()

test_states = 1
trials = 1
time_steps = 10
grid_starter = 32
grid_factors = 1

position_agg_err = np.zeros((grid_factors,test_states,time_steps))
overlap_agg_err = np.zeros((grid_factors,test_states,time_steps))
density_agg_err = np.zeros((grid_factors,test_states,time_steps))
energy_agg_err = np.zeros((grid_factors,test_states,time_steps))

for i in range(test_states):
    for q in range(trials):
        # print(i,q)
        
        traj = generate_pot_TD(time_steps,10)[0]
        true_state_traj = []
        
        ground_state_PES_1 = traj[0]
        temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
        basis_full = temp_full[1][:,0:n_states].T
        
        state_coeff = 2*np.random.random(n_states)-1
        state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
        state_full = np.matmul(state_coeff,basis_full)
        state_full = state_full[0]
        
        for t in range(time_steps):
            hamiltonian = prop_1D(xlist,traj[t],mass)
            delta_t = 1000/24.18    
            prop_full = expm(-1j*hamiltonian*delta_t)
            state_full = np.matmul(state_full,prop_full)
            true_state_traj.append(state_full)
            
        true_state_traj = np.array(true_state_traj)
        
        for f in range(grid_factors):
            
            factor = 2**(f+1)
            # print(grid_starter/factor)
            temp_sparse = fgh_1D(xlist[::factor],ground_state_PES_1[::factor],mass)
            basis_sparse = temp_sparse[1][:,0:n_states].T
            
            for k in range(n_states):
                if (np.sum(temp_full[1][:,k][::factor]*temp_sparse[1][:,k])*np.sqrt(factor)) < 0:
                    temp_sparse[1][:,k] = -1*temp_sparse[1][:,k]
            state_sparse = np.matmul(state_coeff,basis_sparse)
            state_sparse = state_sparse[0]
            
            for t in range(time_steps):
                hamiltonian = prop_1D(xlist[::factor],traj[t][::factor],mass)
                delta_t = 1000/24.18
                prop_sparse = expm(-1j*hamiltonian*delta_t)
                state_sparse = np.matmul(state_sparse,prop_sparse)
                                
                position_agg_err[f,i,t] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(true_state_traj[t])*xlist)))
                overlap_agg_err[f,i,t] += (get_den(np.sum(state_sparse*np.conj(true_state_traj[t][::factor])*np.sqrt(factor))))
                density_agg_err[f,i,t] += np.sum(np.abs((get_den(state_sparse)-get_den(true_state_traj[t][::factor])*factor)))   
                energy_agg_err[f,i,t] += ((np.dot(get_den(np.matmul(state_sparse.T,basis_sparse.T)),temp_full[0][:n_states]))-(np.dot(get_den(np.matmul(true_state_traj[t].T,basis_full.T)),temp_full[0][:n_states])))

                
end = time.time()

print(end-start)

print("\n OVERLAP ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(overlap_agg_err[i,f,time_steps-1]/trials)
      
print("\n DENSITY ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(density_agg_err[i,f,time_steps-1]/trials)

print("\n POSITION ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(position_agg_err[i,f,time_steps-1]/trials)
        
print("\n ENERGY ERRORS")
for i in range((overlap_agg_err).shape[0]):
    print(int(grid_starter/(2**(i+1))))
    for f in range((overlap_agg_err).shape[1]):
        print(energy_agg_err[i,f,time_steps-1]/trials)

# file_overlap_agg_err = '/Users/maximsecor/Desktop/overlap_agg_err.csv'
# file_density_agg_err = '/Users/maximsecor/Desktop/density_agg_err.csv'
# file_position_agg_err = '/Users/maximsecor/Desktop/position_agg_err.csv'

file_overlap_agg_err = 'overlap_agg_err.csv'
file_density_agg_err = 'density_agg_err.csv'
file_position_agg_err = 'position_agg_err.csv'

os.system('touch ' + file_overlap_agg_err)
os.system('touch ' + file_density_agg_err)
os.system('touch ' + file_position_agg_err)

df_overlap_agg_err = pd.DataFrame(overlap_agg_err.reshape(grid_factors*test_states*time_steps))
df_density_agg_err = pd.DataFrame(density_agg_err.reshape(grid_factors*test_states*time_steps))
df_position_agg_err = pd.DataFrame(position_agg_err.reshape(grid_factors*test_states*time_steps))

df_overlap_agg_err.to_csv(file_overlap_agg_err, index = False, header=True)
df_density_agg_err.to_csv(file_density_agg_err, index = False, header=True)
df_position_agg_err.to_csv(file_position_agg_err, index = False, header=True)

#%%

fig = plt.figure(dpi=600)
full_test = generate_pot_TD(1000,10)
traj = full_test[0]
equil = full_test[3]
start = full_test[0][0]

for i in range(100):
    plt.ylim(-25,50)
    plt.plot(traj[i*10]*627,'k',linewidth=0.1)
    # plt.plot(get_den(true_state_traj[5])*500)
    
plt.plot(equil*627,':r',linewidth=2)
plt.plot(start*627,':b',linewidth=2)

#%%

fig = plt.figure(dpi=600)
camera = Camera(fig)

for i in range(10):
# for i in range(5):
    
    tstep = i
    
    # overlap = np.sum(timeseries_state_true[i]*(np.conj(timeseries_state_pred[i])))
    # overlap_squared = np.real(overlap*np.conj(overlap))
    # pos_error = np.abs(timeseries_pos_true[i]-timeseries_pos_pred[i])
    
    plt.text(0.75, 45, "Time Elapsed: "+str(str(int(tstep))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))
    # plt.text(0.75, 41, r'$\langle\Psi^{true}|\Psi^{pred}\rangle$: '+"%.3f" % overlap_squared, color="k", fontsize=10, bbox=dict(facecolor='white'))
    # plt.text(0.75, 37  , r'$|\langle x \rangle_{true}-\langle x \rangle_{pred}|$: '+"%.3f" % pos_error, color="k", fontsize=10, bbox=dict(facecolor='white'))
    
    plt.ylim(-5,50)
    # plt.plot(xlist,dynamic_pot_list[i]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b',xlist,(timeseries_pred[i]-timeseries_true[i])*100,'g')
    plt.plot(xlist,get_den(true_state_traj[i])*500,'b',xlist,traj[i]*627,'k')
    # plt.scatter(timeseries_pos_true[i], test(timeseries_pos_true[i]), marker='o', color='r')
    # plt.scatter(timeseries_pos_pred[i], test(timeseries_pos_pred[i]), marker='o', color='b')
    
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
    
animation = camera.animate(interval=70)
animation.save('/Users/maximsecor/Desktop/potentials.gif', writer = 'imagemagick')

#%%

time_steps = 5000
grid_size  = 64
traj = generate_pot_TD(time_steps,10)[0]
xlist = np.linspace(-1.5,1.5,grid_size)*1.8897
n_states = 10
true_state_traj = []

ground_state_PES_1 = traj[0]
temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
basis_full = temp_full[1][:,0:n_states].T

state_coeff = 2*np.random.random(n_states)-1
state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
state_full = np.matmul(state_coeff,basis_full)
state_full = state_full[0]
state_test = state_full

true_state_traj = []
for t in range(time_steps):
    hamiltonian = prop_1D(xlist,traj[t],mass)
    delta_t = 1000/24.18    
    prop_full = expm(-1j*hamiltonian*delta_t)
    state_full = np.matmul(state_full,prop_full)
    true_state_traj.append(state_full) 
true_state_traj = np.array(true_state_traj)


test_state_traj = []
for t in range(time_steps):
    hamiltonian = prop_1D(xlist,traj[t]+np.random.random(),mass)
    delta_t = 1000/24.18    
    prop_full = expm(-1j*hamiltonian*delta_t)
    state_test = np.matmul(state_test,prop_full)
    test_state_traj.append(state_test) 
test_state_traj = np.array(test_state_traj)

#%%

plt.plot(get_den(true_state_traj[4000]))
plt.plot(get_den(test_state_traj[4000])+0.1)

#%%

print(true_state_traj[4000]-test_state_traj[4000])

#%%

print(get_den(np.sum(true_state_traj[4000]*np.conj(test_state_traj[4000]))))

#%%

position_agg_err[f,i,t] += np.abs((np.sum(get_den(state_sparse)*xlist[::factor])-np.sum(get_den(true_state_traj[t])*xlist)))
overlap_agg_err[f,i,t] += (get_den(np.sum(state_sparse*np.conj(true_state_traj[t][::factor])*np.sqrt(factor))))
density_agg_err[f,i,t] += np.sum(np.abs((get_den(state_sparse)-get_den(true_state_traj[t][::factor])*factor)))   
energy_agg_err[f,i,t] += ((np.dot(get_den(np.matmul(state_sparse.T,basis_sparse.T)),temp_full[0][:n_states]))-(np.dot(get_den(np.matmul(true_state_traj[t].T,basis_full.T)),temp_full[0][:n_states])))

#%%



        