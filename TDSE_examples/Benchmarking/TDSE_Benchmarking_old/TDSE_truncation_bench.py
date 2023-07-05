#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:20:27 2021

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

dt = 1000

def generate_propagator_exact(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop

def generate_propagator_first(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t
    
    return prop_s

def generate_propagator_second(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t-0.5*np.matmul(hamiltonian,hamiltonian)*delta_t**2
    
    return prop_s

def generate_propagator_third(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t-0.5*np.matmul(hamiltonian,hamiltonian)*delta_t**2+(1j/6)*np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian))*delta_t**3
    
    return prop_s

def generate_propagator_fourth(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t-0.5*np.matmul(hamiltonian,hamiltonian)*delta_t**2+(1j/6)*np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian))*delta_t**3+(1/24)*np.matmul(hamiltonian,np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian)))*delta_t**4
    
    return prop_s

def generate_propagator_fifth(test_pot_2):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = dt/24.18
    prop_s = np.eye(len(hamiltonian))-1j*hamiltonian*delta_t-0.5*np.matmul(hamiltonian,hamiltonian)*delta_t**2+(1j/6)*np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian))*delta_t**3+(1/24)*np.matmul(hamiltonian,np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian)))*delta_t**4-(1j/120)*np.matmul(hamiltonian,np.matmul(hamiltonian,np.matmul(hamiltonian,np.matmul(hamiltonian,hamiltonian))))*delta_t**5
    
    return prop_s

#%%
    
import math 

def generate_propagator_nth(test_pot_2,time_step,n_order):
    
    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
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
        
        # print((1/(math.factorial(i))),((-1j)**i))
        
        prop_s = prop_s + prop_add
            
    return prop_s


#%%

test = generate_propagator_nth(test_pot,1,2)


#%%
    
grid_size = 32
mass = 1836

test_pot = generate_pot()
plt.ylim(-1,50)
plt.plot(test_pot*627)
plt.show()

n_states = 5

#%%

print(generate_propagator_fourth(test_pot))
print(generate_propagator_nth(test_pot,1,4))

#%%

test_basis = generate_basis(test_pot,n_states)
test_state = generate_state(test_basis,n_states)

test_state_exact = test_state
test_state_first = test_state
test_state_second = test_state
test_state_third = test_state
test_state_fourth = test_state
test_state_fifth = test_state
test_state_10 = test_state

test_prop_exact = generate_propagator_exact(test_pot)
test_prop_first = generate_propagator_first(test_pot)
test_prop_second = generate_propagator_second(test_pot)
test_prop_third = generate_propagator_third(test_pot)
test_prop_fourth = generate_propagator_fourth(test_pot)
test_prop_fifth = generate_propagator_fifth(test_pot)
test_prop_10 = (generate_propagator_nth(test_pot,dt,64))

#%%

print(np.max(get_den(test_prop_exact)))
print(np.max(get_den(test_prop_first)))
print(np.max(get_den(test_prop_second)))
print(np.max(get_den(test_prop_third)))
print(np.max(get_den(test_prop_fourth)))
print(np.max(get_den(test_prop_fifth)))
print(np.max(get_den(test_prop_10)))

#%%

spans = np.array([1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000])

for j in range(len(spans)):
    for i in range(spans[j]):
        
        test_state_first = np.matmul(test_state_first,test_prop_first)
        test_state_second = np.matmul(test_state_second,test_prop_second)
        test_state_third = np.matmul(test_state_third,test_prop_third)
        test_state_fourth = np.matmul(test_state_fourth,test_prop_fourth)
        # test_state_fifth = np.matmul(test_state_fifth,test_prop_fifth)
        test_state_10 = np.matmul(test_state_10,test_prop_10)
        test_state_exact = np.matmul(test_state_exact,test_prop_exact)
        
    plt.ylim(-1,50)
    plt.title(str(spans[j]))
    plt.plot(test_pot*627)
    plt.plot(get_den(test_state_first)*100+25)
    plt.plot(get_den(test_state_second)*100+20)
    plt.plot(get_den(test_state_third)*100+15)
    plt.plot(get_den(test_state_fourth)*100+10)
    # plt.plot(get_den(test_state_fifth)*100+5)
    plt.plot(get_den(test_state_10)*100+5)
    plt.plot(get_den(test_state_exact)*100)
    plt.show()
    
#%%
   
def generate_propagator_exact(test_pot_2,time_step):

    xlist = np.linspace(-0.75,0.75,grid_size)*1.8897
    hamiltonian = prop_1D(xlist,test_pot_2,mass)
    delta_t = time_step/24.18
    prop = expm(-1j*hamiltonian*delta_t)
    
    return prop
    
#%%

# spans = np.array([1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000])
spans = np.array([1,3,10,30,100,300,1000,3000,10000])
orders = np.linspace(1,64,64,dtype=int)

test_basis = generate_basis(test_pot,n_states)
test_state = generate_state(test_basis,n_states)

for j in range(len(spans)):
    for i in range(len(orders)):
        
# for j in range(5):
#     for i in range(5):
        
        test_prop_10 = (generate_propagator_nth(test_pot,spans[j],orders[i]))
        test_prop_exact = generate_propagator_exact(test_pot,spans[j])

        test_state_10 = np.matmul(test_state,test_prop_10)
        test_state_exact = np.matmul(test_state,test_prop_exact)
        
        print(spans[j],orders[i],get_den(np.sum(test_state_10*np.conj(test_state_exact))))

#%%
    
# spans = np.array([1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000])
# spans = np.array([1,3,10,30,100,300,1000,3000,10000])
spans = np.array([1,3,10,30,100,300])
# orders = np.linspace(1,64,64,dtype=int)
orders = np.array([1,2,4,8,12,16,20,24,28,32,48,64,128,256])
sim_length = np.array([1,3,10,30,100,300,1000,3000,10000])

test_basis = generate_basis(test_pot,n_states)
test_state = generate_state(test_basis,n_states)

tracker_data = np.zeros((len(spans),len(orders),len(sim_length)))

for j in range(len(spans)):
    for i in range(len(orders)):
        
# for j in range(5):
#     for i in range(5):
        
        test_prop_10 = (generate_propagator_nth(test_pot,spans[j],orders[i]))
        test_prop_exact = generate_propagator_exact(test_pot,spans[j])

        for k in range(len(sim_length)):
        # for k in range(5):

            test_state_10 = test_state
            test_state_exact = test_state
            
            for t in range(sim_length[k]):

                test_state_10 = np.matmul(test_state_10,test_prop_10)
                test_state_exact = np.matmul(test_state_exact,test_prop_exact)
            
            tracker_data[j,i,k] = get_den(np.sum(test_state_10*np.conj(test_state_exact)))
            
            if str(tracker_data[j,i,k]) == str(math.nan):
                print(spans[j],orders[i],sim_length[k]," DOES NOT WORK")
            else:
            
                checker = np.abs(tracker_data[j,i,k] - 1)
                
                if checker > 0.1:
                    print(spans[j],orders[i],sim_length[k]," DOES NOT WORK")
                else:
                    print(spans[j],orders[i],sim_length[k]," WORKS")

#%%
          
start = time.time()            

trials = 10

orders = np.linspace(1,128,6,dtype=int)

print(orders)

truncation_results = np.zeros((len(orders)))


for q in range(trials):
    
    print(q)            
    
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
    
    test_basis = generate_basis(test_pot,n_states)
    test_state = generate_state(test_basis,n_states)
    
    tracker_data = np.zeros((len(orders)))
    
    for i in range(len(orders)):
        
        test_state_10 = test_state
        test_state_exact = test_state
        
        for t in range(1000):
            
            test_prop_10 = (generate_propagator_nth(traj[t],1000,orders[i]))
            test_prop_exact = generate_propagator_exact(traj[t],1000)
    
            test_state_10 = np.matmul(test_state_10,test_prop_10)
            test_state_exact = np.matmul(test_state_exact,test_prop_exact)
        
        tracker_data[i] = get_den(np.sum(test_state_10*np.conj(test_state_exact)))
        
        if str(tracker_data[i]) != str(math.nan):

            checker = np.abs(tracker_data[i] - 1)
            
            if checker < 0.1:

                truncation_results[i] = truncation_results[i]+1
                
end = time.time() 

print(end-start)
                                
plt.plot(truncation_results)
print(truncation_results)

#%%

start = time.time()            

trials = 100

orders = np.linspace(1,128,128,dtype=int)

print(orders)

truncation_results = np.zeros((len(orders)))


for q in range(trials):
    
    print(q)            
    
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

    traj = []
    
    for t in range(10000):
        
        traj.append(ground_state_PES_1)
    
    traj = np.array(traj)
    
    test_basis = generate_basis(test_pot,n_states)
    test_state = generate_state(test_basis,n_states)
    
    tracker_data = np.zeros((len(orders)))
    
    for i in range(len(orders)):
        
        test_state_10 = test_state
        test_state_exact = test_state
        
        test_prop_10 = (generate_propagator_nth(traj[t],100,orders[i]))
        test_prop_exact = generate_propagator_exact(traj[t],100)
        
        for t in range(1):
    
            test_state_10 = np.matmul(test_state_10,test_prop_10)
            test_state_exact = np.matmul(test_state_exact,test_prop_exact)
        
        tracker_data[i] = get_den(np.sum(test_state_10*np.conj(test_state_exact)))
        
        if str(tracker_data[i]) != str(math.nan):

            checker = np.abs(tracker_data[i] - 1)
            
            if checker < 0.1:

                truncation_results[i] = truncation_results[i]+1
                
end = time.time() 

print(end-start)
                                
plt.plot(truncation_results)
print(truncation_results)


#%%

# prot_step = np.array([1,2.5,10,25,100,250,1000])

factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000])
results = np.zeros((len(factors)))

for q in range(10):

    start = time.time()
    
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
    dt_da = 0.001/0.024188
    
    f0 = 55.734
    mu = 0.2652
    lam = (5*np.random.random())
    x_solv = np.sqrt(lam/(2*f0))
    freq_solv = np.sqrt(f0/mu)
    dt_solv = 0.000001
    
    traj = []
    
    for t in range(100000):
        
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
    test_basis = generate_basis(test_pot,n_states)
    test_state = generate_state(test_basis,n_states)
    
    # factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000,25000,50000])
    # factors = np.array([1,2,5,10,25,50,100,250,500,1000,2500,5000,10000])
    
    for f in range(len(factors)):
    
        test_state_exact = test_state
        
        for t in range(int(100000/factors[f])):
            
            test_prop_exact = generate_propagator_exact(traj[t*factors[f]],factors[f])
            test_state_exact = np.matmul(test_state_exact,test_prop_exact)
    
        final_states.append(test_state_exact)
    
    end = time.time()
    
    print(end-start)
    
    final_states = np.array(final_states)
    
    # plt.ylim(-1,50)
    # plt.plot(traj[0]*627)
    # plt.plot(get_den(final_states[0])*100+1)
    # plt.plot(get_den(final_states[1])*100+6)
    # plt.plot(get_den(final_states[2])*100+11)
    # plt.plot(get_den(final_states[3])*100+16)
    # plt.plot(get_den(final_states[4])*100+21)
    # plt.show()
    
    for i in range(len(factors)):
        results[i] = results[i] + (np.sqrt(get_den(np.sum(final_states[0]*np.conj(final_states[i])))))
    
    # plt.plot(get_den(test_state))
    # plt.show()
    
#%%

for i in range(len(factors)):
    print(results[i]/10)