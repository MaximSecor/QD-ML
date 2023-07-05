#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:05:21 2021

@author: maximsecor
"""

import numpy as np
import matplotlib.pyplot as plt


#%%

a_file = open("/Users/maximsecor/Desktop/test_7.txt", "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

print(list_of_lists)

#%%

list_of_lists = np.array(list_of_lists)

#%%


for i in range(int(len(list_of_lists[1::2]))):
    print(list_of_lists[1::2][i][1])


#%%


temp_list = []
for i in range(int(len(list_of_lists)/6)):
    
    temp = 0
    for j in range(5):
        temp = temp + float(list_of_lists[i+j*1000][1])
    temp_list.append(temp/5)

temp_list = np.array(temp_list)

#%%

for i in range(200):
    print(temp_list[i+800])

#%%


test_plot = []
for i in range(len(list_of_lists)-4):
    print(list_of_lists[i][1])
    test_plot.append(float(list_of_lists[i][1]))
test_plot = np.array(test_plot)

plt.ylim(0.99,1.001)
plt.plot(test_plot)