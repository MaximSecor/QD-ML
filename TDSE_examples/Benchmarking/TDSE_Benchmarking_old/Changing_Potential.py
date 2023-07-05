#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:00:51 2021

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

for i in range(6):

    en_hist = []
    trials = 10000
    
    for q in range(trials):
        
        # print(q)
        
        grid_size = 32
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
        
        n_states = i
        
        temp_full = fgh_1D(xlist,ground_state_PES_1,mass)
        basis_full = temp_full[1][:,0:n_states].T
        
        
        state_coeff = 2*np.random.random(n_states)-1
        state_coeff = state_coeff / np.sqrt(np.sum(state_coeff**2)).reshape(1,-1)
        
        state_full = np.matmul(state_coeff,basis_full)
        state_full = state_full[0]
        
        en_hist.append(np.matmul(state_coeff**2,temp_full[0][:i]))
    
    
    
    en_hist = np.array(en_hist)*627
    plt.hist(en_hist, bins=100)
    plt.show()

#%%

eps_not = 79.2
eps_inf = 4.2

tau0 = 0.0103
tauD = 8.72
taul = (eps_inf/eps_not)*tauD
taul0 = (eps_inf/eps_not)*tau0

f0 = 55.734
boltzkcal = 0.001986
temp = 300

gamma = (tau0+tauD)/(tau0*tauD)
sigma = np.sqrt(2*boltzkcal*temp*(taul0+taul)*(1/f0))/(tau0+taul)

ksi = np.random.normal(0,1)
eta = np.random.normal(0,1)

dE = (20*np.random.random()-10)/627
X0 = (0.5*np.random.random())
red_mass = ((12*12)/(12+12))*1836

coord = np.array([X0,dE])
veloc = np.array([0,0])
accel = np.array([0,0])
k_const = np.array([0.0451,0.0451])
coord_eq = np.array([0,0])

dt = 1/0.0241888
traj = []

dt2 = dt**2
sqdt = np.sqrt(dt)
dt32 = dt*sqdt

for t in range(100):
    coord = coord + veloc*dt + 0.5*accel*dt**2
    accel_old = accel
    accel = -k_const*coord/red_mass
    veloc = veloc + 0.5*(accel_old+accel)*dt
    traj.append(coord[1])
    
    # print(np.sum(k_const*coord**2)*0.5+np.sum(veloc**2)*0.5)
    
plt.plot(traj)

#%%

test = []
for i in range(1000000):
    test.append(np.random.normal(2,1))
test = np.array(test)
plt.hist(test, bins=50)

print(np.mean(test))
print(np.std(test))


#%%

"water"
eps_not = 79.2
eps_inf = 4.2
f0 = 55.734
tau0 = 0.0103
tauD = 8.72
lam = 10

taul = (eps_inf/eps_not)*tauD
taul0 = (eps_inf/eps_not)*tau0

boltzkcal = 0.001986
temp = 300

gamma = (tau0+tauD)/(tau0*tauD)
sigma = np.sqrt(2*boltzkcal*temp*(taul0+taul)*(1/f0))/(tau0+taul)

dt = 0.001
sq3 = np.sqrt(3)
dt2 = dt**2
sqdt = np.sqrt(dt)
dt32 = dt*sqdt

x = 0.00001
v = 0.00001

traj = []

for t in range(10000000):
    
    # traj.append(100*x*np.sqrt(2*f0*lam))
    traj.append(x)
    
    ksi = np.random.normal(0,1)
    eta = np.random.normal(0,1)
    
    fx = -x/(tau0*taul)
    vhalf = v + 0.5*dt*fx -0.5*dt*gamma*v + 0.5*sqdt*sigma*ksi - 0.125*dt2*gamma*(fx-gamma*v) - 0.25*dt32*gamma*sigma*(0.5*ksi+eta/sq3)
    x = x + dt*vhalf + 0.5*dt32*sigma*eta/sq3
    fx = -x/(tau0*taul)
    v = vhalf + 0.5*dt*fx -0.5*dt*gamma*vhalf + 0.5*sqdt*sigma*ksi - 0.125*dt2*gamma*(fx-gamma*vhalf) - 0.25*dt32*gamma*sigma*(0.5*ksi+eta/sq3)

    # en = 0.5*f0*tau0*taul*v*v + 0.5*f0*x*x
    # print(en)
    
plt.plot(traj)

#%%

traj = np.array(traj)
x_2 = (np.mean(traj**2))

#%%

cor_xtxt = []

for i in range(1000):
    
    i = i*10
    correlation = 0
    
    for j in range(len(traj)-i):
        
        correlation = correlation + (traj[j]*traj[j+i])/(x_2)

    print(correlation/(len(traj)-i))
    cor_xtxt.append((correlation/(len(traj)-i)))

#%%



import scipy

cor_xtxt = np.array(cor_xtxt)
cor_xtxt_fft = scipy.fft(cor_xtxt)
cor_xtxt_fft = np.concatenate((cor_xtxt_fft[500:],cor_xtxt_fft[:500]))
plt.plot(cor_xtxt_fft)








