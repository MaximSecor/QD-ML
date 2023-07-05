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

test_states = 5
trials = 5
time_steps = 10
grid_starter = 1024
grid_factors = 7

position_agg_err = np.zeros((grid_factors,test_states,time_steps))
overlap_agg_err = np.zeros((grid_factors,test_states,time_steps))
density_agg_err = np.zeros((grid_factors,test_states,time_steps))
energy_agg_err = np.zeros((grid_factors,test_states,time_steps))

for i in range(test_states):
    for q in range(trials):
        # print(i,q)
        
        n_states = i + 1
        grid_size = grid_starter
        mass = 1836
        
        xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
        KX_1 = (2500*np.random.random()+1500)/(350*627)
        KX_2 = (2500*np.random.random()+1500)/(350*627)
        dE = (10*np.random.random()-5)/627
        
        red_mass = ((12*12)/(12+12))*1836
        coord_eq = 0.25*np.random.random()
        X0 = 2*X0*np.random.random()
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
        
        for t in range(1000):
            
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

# plt.ylim(-1,50)
# plt.plot(traj[5]*627)
# plt.plot(get_den(true_state_traj[5])*500)

#%%

# fig = plt.figure(dpi=600)
# camera = Camera(fig)

# for i in range(1000):
# # for i in range(5):
    
#     tstep = i
    
#     # overlap = np.sum(timeseries_state_true[i]*(np.conj(timeseries_state_pred[i])))
#     # overlap_squared = np.real(overlap*np.conj(overlap))
#     # pos_error = np.abs(timeseries_pos_true[i]-timeseries_pos_pred[i])
    
#     plt.text(0.75, 45, "Time Elapsed: "+str(str(int(tstep))+" fs"), color="k", fontsize=10, bbox=dict(facecolor='white'))
#     # plt.text(0.75, 41, r'$\langle\Psi^{true}|\Psi^{pred}\rangle$: '+"%.3f" % overlap_squared, color="k", fontsize=10, bbox=dict(facecolor='white'))
#     # plt.text(0.75, 37  , r'$|\langle x \rangle_{true}-\langle x \rangle_{pred}|$: '+"%.3f" % pos_error, color="k", fontsize=10, bbox=dict(facecolor='white'))
    
#     plt.ylim(-5,50)
#     # plt.plot(xlist,dynamic_pot_list[i]*627,'k',xlist,timeseries_true[i]*100,'r',xlist,timeseries_pred[i]*100,'b',xlist,(timeseries_pred[i]-timeseries_true[i])*100,'g')
#     plt.plot(xlist,get_den(true_state_traj[i])*500,'b',xlist,traj[i]*627,'k')
#     # plt.scatter(timeseries_pos_true[i], test(timeseries_pos_true[i]), marker='o', color='r')
#     # plt.scatter(timeseries_pos_pred[i], test(timeseries_pos_pred[i]), marker='o', color='b')
    
#     plt.xlabel('x (bohr)')
#     plt.ylabel('Energy (kcal/mol)')
#     # plt.title('Constantly Changing Hamiltonian')
    
#     plt.rc('font', family='Helvetica')
    
#     SMALL_SIZE = 10
#     MEDIUM_SIZE = 12
#     BIGGER_SIZE = 14
    
#     plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
#     camera.snap()
    
# animation = camera.animate(interval=70)
# animation.save('/Users/maximsecor/Desktop/potentials.gif', writer = 'imagemagick')


        