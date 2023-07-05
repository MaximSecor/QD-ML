#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:18:41 2021

@author: maximsecor
"""

import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(precision=8,suppress=True)

#%%

temp_name = "/Users/maximsecor/Desktop/TIT_1000.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    title = (list_of_lists[i][0].split('/')[0])
    layers = title.split('_')[1]
    nodes = title.split('_')[2]
    rate = title.split('_')[3]
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    
    data = np.array([layers, nodes, rate, amplitude_error, phase_error],dtype="float")
    data_list.append(data)
    
data_list = np.array(data_list)

print(data_list)

#%%

amp_max = np.argmax(data_list[:,3])
print(amp_min, data_list[amp_max])

phs_min = np.argmin(data_list[:,4])
print(phs_min, data_list[phs_min])

#%%

temp_name = "/Users/maximsecor/Desktop/TIT_Series.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    
    print(amplitude_error,phase_error)
    data = np.array([amplitude_error, phase_error],dtype="float")
    
    data_list.append(data)
    
data_list = np.array(data_list)

plt.ylim(0.999,1.001)
plt.plot(data_list[:,0])
plt.show()

plt.plot(data_list[:,1])
plt.show()

#%%

temp_name = "/Users/maximsecor/Desktop/TIT_Pots.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    data = np.array([amplitude_error, phase_error],dtype="float")
    data_list.append(data)
    
data_list = np.array(data_list)
print(np.mean(data_list,0))

#%%

temp_name = "/Users/maximsecor/Desktop/TIT_Times.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    title = (list_of_lists[i][0].split('/')[0])
    step_size = (title.split('_')[1])
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    data = np.array([step_size,amplitude_error, phase_error],dtype="float")
    # print(data)
    data_list.append(data)
    
data_list = np.array(data_list)
# print(data_list)
print(data_list[data_list[:,0].argsort()[::1]])

#%%

temp_name = "/Users/maximsecor/Desktop/TIT_Steps.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    title = (list_of_lists[i][0].split('/')[0])
    step_size = (title.split('_')[1])
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    data = np.array([step_size,amplitude_error, phase_error],dtype="float")
    # print(data)
    data_list.append(data)
    
data_list = np.array(data_list)
print(np.concatenate((data_list[30:], data_list[:30])))
# print(data_list[data_list[:,0].argsort()[::1]])

#%%
    
temp_name = "/Users/maximsecor/Desktop/TIT_Error.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

list_of_lists_1 = (list_of_lists[1::6])
list_of_lists_2 = (list_of_lists[3::6])
list_of_lists_3 = (list_of_lists[5::6])

print(list_of_lists_3[0])
# print(list_of_lists_1[0][-1])

#%%

data_list = []

for i in range(len(list_of_lists_1)):
    
    title = (list_of_lists_1[i][0].split('/'))
    layers = title[0].split('_')[1]
    nodes = title[0].split('_')[2]
    rate = title[0].split('_')[3]
    
    error = float(list_of_lists_1[i][-1].split(']')[0])
    
    data = np.array([layers, nodes, rate, error],dtype="float")
    # print(data)
    data_list.append(data)
    
data_list_1 = np.array(data_list)
print(data_list_1)

#%%

data_list = []

for i in range(len(list_of_lists_2)):
    
    title = (list_of_lists_2[i][0].split('/'))
    layers = title[0].split('_')[1]
    nodes = title[0].split('_')[2]
    rate = title[0].split('_')[3]
    
    error = float(list_of_lists_2[i][-1].split(']')[0])
    
    data = np.array([layers, nodes, rate, error],dtype="float")
    # print(data)
    data_list.append(data)
    
data_list_2 = np.array(data_list)
print(data_list_2)

#%%

data_list = []

for i in range(len(list_of_lists_3)):
    
    title = (list_of_lists_3[i][0].split('/'))
    layers = title[0].split('_')[1]
    nodes = title[0].split('_')[2]
    rate = title[0].split('_')[3]
        
    phs_error = float(list_of_lists_3[i][-1].split(']')[0])
    amp_error = float(list_of_lists_3[i][-2].split('[')[1])
    
    data = np.array([layers, nodes, rate, amp_error, phs_error],dtype="float")
    # print(data)
    data_list.append(data)
    
data_list_3 = np.array(data_list)
print(data_list_3)

#%%

amp_max = np.argmin(data_list_1[:,3])
print(amp_max, data_list_1[amp_max])

amp_max = np.argmin(data_list_2[:,3])
print(amp_max, data_list_2[amp_max])

amp_max = np.argmax(data_list_3[:,3])
print(amp_max, data_list_3[amp_max])

amp_max = np.argmin(data_list_3[:,4])
print(amp_max, data_list_3[amp_max])

#%%

temp_name = "/Users/maximsecor/Desktop/TDSE_1000.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

idx = 0

data_list = []

for i in range(len(list_of_lists)):
    
    # print(idx%2)
    # idx += 1
    
    title = (list_of_lists[i][0].split('/'))
    
    time_step = title[0].split('_')[1]
    layers = title[1].split('_')[1]
    nodes = title[1].split('_')[2]
    rate = title[1].split('_')[3]
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    
    if time_step == '05':
        time_step = 0.5
    
    data = np.array([time_step, layers, nodes, rate, amplitude_error, phase_error],dtype="float")
    data_list.append(data)
    
data_list = np.array(data_list)

time_dep_pot = data_list[0::2]
time_indep_pot = data_list[1::2]
print(np.max(time_dep_pot[:,4]))
print(np.max(time_indep_pot[:,4]))

#%%

steps = [0.5,1,2,3,5,10]

for j in range(6):
    
    test_1 = []
    
    for i in range(len(time_dep_pot)):
        if (steps[j]==time_dep_pot[i][0]):
            test_1.append(time_dep_pot[i])
    
    test_1 = np.array(test_1)

    amp_min = np.argmax(test_1[:,4])
    print(test_1[amp_min,1:])
    # phs_min = np.argmin(test_1[:,5])
    # print(phs_min, test_1[phs_min])
    # print("\n")

#%%

steps = [0.5,1,2,3,5,10]

for j in range(6):
    
    test_1 = []
    
    for i in range(len(time_indep_pot)):
        if (steps[j]==time_indep_pot[i][0]):
            test_1.append(time_indep_pot[i])
    
    test_1 = np.array(test_1)

    amp_min = np.argmax(test_1[:,4])
    print(test_1[amp_min,1:])
    # phs_min = np.argmin(test_1[:,5])
    # print(phs_min, test_1[phs_min])
    # print("\n")

#%%
    
temp_name = "/Users/maximsecor/Desktop/Errors.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

list_of_lists_1 = (list_of_lists[1::6])
list_of_lists_2 = (list_of_lists[3::6])
list_of_lists_3 = (list_of_lists[5::6])

print(list_of_lists_3[0])
# print(list_of_lists_1[0][-1])

#%%

data_list = []

for i in range(len(list_of_lists_1)):
    
    title = (list_of_lists_1[i][0].split('/'))
    
    time_step = title[0].split('_')[1]
    layers = title[1].split('_')[1]
    nodes = title[1].split('_')[2]
    rate = title[1].split('_')[3]
    
    if time_step == '05':
        time_step = 0.5
        
    error = float(list_of_lists_1[i][-1].split(']')[0])
    
    data = np.array([time_step, layers, nodes, rate, error],dtype="float")
    print(data)
    data_list.append(data)
    
data_list_1 = np.array(data_list)

#%%

data_list = []

for i in range(len(list_of_lists_2)):
    
    title = (list_of_lists_2[i][0].split('/'))
    
    time_step = title[0].split('_')[1]
    layers = title[1].split('_')[1]
    nodes = title[1].split('_')[2]
    rate = title[1].split('_')[3]
    
    if time_step == '05':
        time_step = 0.5
        
    error = float(list_of_lists_2[i][-1].split(']')[0])
    
    data = np.array([time_step, layers, nodes, rate, error],dtype="float")
    print(data)
    data_list.append(data)
    
data_list_2 = np.array(data_list)

#%%

data_list = []

for i in range(len(list_of_lists_3)):
    
    title = (list_of_lists_3[i][0].split('/'))
    
    time_step = title[0].split('_')[1]
    layers = title[1].split('_')[1]
    nodes = title[1].split('_')[2]
    rate = title[1].split('_')[3]
    
    if time_step == '05':
        time_step = 0.5
        
    phs_error = float(list_of_lists_3[i][-1].split(']')[0])
    amp_error = float(list_of_lists_3[i][-2].split('[')[1])
    
    data = np.array([time_step, layers, nodes, rate, amp_error,phs_error],dtype="float")
    print(data)
    data_list.append(data)
    
data_list_3 = np.array(data_list)

#%%

steps = [0.5,1,2,3,5,10]

for j in range(6):
    
    test_1 = []
    
    for i in range(len(data_list_1)):
        if (steps[j]==data_list_1[i][0]):
            test_1.append(data_list_1[i])
    
    test_1 = np.array(test_1)

    amp_min = np.argmin(test_1[:,4])
    print(amp_min, test_1[amp_min])
    # phs_min = np.argmin(test_1[:,5])
    # print(phs_min, test_1[phs_min])
    # print("\n")

#%%

steps = [0.5,1,2,3,5,10]

for j in range(6):
    
    test_1 = []
    
    for i in range(len(data_list_2)):
        if (steps[j]==data_list_2[i][0]):
            test_1.append(data_list_2[i])
    
    test_1 = np.array(test_1)

    amp_min = np.argmin(test_1[:,4])
    print(amp_min, test_1[amp_min])
    # phs_min = np.argmin(test_1[:,5])
    # print(phs_min, test_1[phs_min])
    # print("\n")

#%%

steps = [0.5,1,2,3,5,10]

for j in range(6):
    
    test_1 = []
    
    for i in range(len(data_list_3)):
        if (steps[j]==data_list_3[i][0]):
            test_1.append(data_list_3[i])
    
    test_1 = np.array(test_1)

    amp_min = np.argmax(test_1[:,4])
    print(amp_min, test_1[amp_min])
    # phs_min = np.argmin(test_1[:,5])
    # print(phs_min, test_1[phs_min])
    # print("\n")




#%%

temp_name = "/Users/maximsecor/Desktop/TDH_series.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists[0])

#%%

data_list = []

for i in range(len(list_of_lists)):
    
    amplitude_error = (list_of_lists[i][3].split("[")[1])
    phase_error = (list_of_lists[i][4].split("]")[0])
    
    print(amplitude_error,phase_error)
    data = np.array([amplitude_error, phase_error],dtype="float")
    
    data_list.append(data)
    
data_list = np.array(data_list)

plt.plot(data_list[:1000,0])
plt.show()

plt.plot(data_list[:1000,1])
plt.show()

plt.plot(data_list[1000:,0])
plt.show()

plt.plot(data_list[1000:,1])
plt.show()

#%%

plt.plot(data_list[:100,0])
plt.plot(data_list[1000:1100,0])
# plt.show()
plt.savefig("/Users/maximsecor/Desktop/plot_1.tif",dpi=600)

#%%

plt.plot(data_list[:100,1])
plt.plot(data_list[1000:1100,1])
# plt.show()
plt.savefig("/Users/maximsecor/Desktop/plot_2.tif",dpi=600)



