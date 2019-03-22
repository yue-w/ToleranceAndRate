# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:48:35 2019

@author: wyue
Optimizing production rate. Clutch Case Study
"""

#import packages
import numpy as np

#import helper functions
import helpers as hp

#Variables to be optimized
r = np.array([1, 1, 1])
k = np.array([3,3,3])


#number of components
m = 3
#Upper specification limit
USp = np.radians(1)
#Nominal value of Y
miuY = np.radians(7.0124)

A = np.array([1, 2, 3])
B = np.array([1, 2, 3])
E = np.array([1, 2, 3])
F = np.array([1, 2, 3])

miu= np.array([55.291,22.86,101.6])


C = hp.C(A,B,r)
sigma = hp.sigma(E,F,r)

epsilon = 1e-7

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D2,D2,D3])

sigmaY = hp.sigmaY(sigma,D)

for ri in r:
    ri_add_epsilon = r
    ri_minus_epsilon = r
    ri_add_epsilon[i] +=  epsilon
    ri_add_epsilon[i] -=  epsilon
    dec_estmate = hp.U(C,USp,sigma,sigmaY,k) 
    
    
    
    