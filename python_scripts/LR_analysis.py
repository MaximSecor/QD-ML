#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 05:48:39 2021

@author: maximsecor
"""


import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=8,suppress=True)

#%%

temp_name = "/Users/maximsecor/Desktop/TDSE_FRESH/LR_output_2b.txt"

a_file = open(temp_name, "r")

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

list_of_lists = np.array(list_of_lists)

print(list_of_lists.shape)

train_data = []
for i in range(100):
    train_data.append(float(list_of_lists[i+1][0]))
train_data = np.array(train_data)

val_data = []
for i in range(100):
    val_data.append(float(list_of_lists[i+1][1]))
val_data = np.array(val_data)

dom = (np.linspace(1,10000,100))

plt.plot(dom,train_data)
plt.plot(dom,val_data)

