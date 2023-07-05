#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:49:09 2021

@author: maximsecor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

file_present = '/Users/maximsecor/Desktop/ANN_2_LC.csv'
data_present = pd.read_csv(file_present)
present = data_present.values.T

#%%

print(present)

#%%

idx = present[0]*10**4
train = present[2]
val = present[1]

#%%

print(idx)

#%%

plt.plot(idx,train)
plt.plot(idx,val)

#%%

plt.plot(idx[20:],train[20:])
plt.plot(idx[20:],val[20:])


