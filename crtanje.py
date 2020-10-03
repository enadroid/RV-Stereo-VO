#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:41:14 2020

@author: ena
"""

from matplotlib import pyplot as plt
from scipy.signal import lfilter
import numpy as np

from ground_truth_data import data

from calculated_data import real_data

ground_truth_traj = data()
calc_coordinates = real_data()      #OVO SU DIREKTNO UNESENI SNIMLJENI NIZOVI, RADI PREGLEDNOSTI U ZASEBNIM FILE-OVIMA

X = []
Y = []

X_g = []
Y_g = []


nula_g = ground_truth_traj[0]
nula_s = calc_coordinates[0]

greska = []

for i in range(len(calc_coordinates)):
    x = 0.95*(calc_coordinates[i][1]-nula_s[1])
    X.append(x)
    y = -0.98*(calc_coordinates[i][0]-nula_s[0])
    Y.append(y)
    
    x_g = ground_truth_traj[i][0]-nula_g[0]
    X_g.append(x_g)
    y_g = ground_truth_traj[i][1]-nula_g[1]
    Y_g.append(y_g)
   
    greska.append(0.05*np.sqrt((x-x_g)**2+(y-y_g)**2))


n = 20  
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,Y)

xx = lfilter(b,a,X)

gg = lfilter(b,a,greska)
    
arg_greska = np.linspace(0,X[-1], len(greska))
fig = plt.figure()
ax = fig.add_subplot(111)
line1 = ax.scatter(xx,yy, c = 'b', s = 4, marker='o')
line2 = ax.scatter(X_g,Y_g, c = 'r', s = 4, marker='o')
ax.legend((line1,line2),('Dobivena trajektorija','Stvarna trajektorija'))
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Trajektorija kojom se kretao automobil')

plt.show()

fig_greska = plt.figure()
ax = fig_greska.add_subplot(111)
ax.scatter(arg_greska,gg, c = 'b', s = 4, marker='o')
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Odstupanje dobivene u odnosu na stvarnu trajektoriju')

plt.show()
    