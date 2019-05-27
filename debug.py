# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:04:08 2019

@author: wyue
"""

#import packages
import numpy as np
import helpers as hp



epsilon = 1e-7

rMin = 1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
rMax= 100
rRst = np.array([20, 20, 20])

A = np.array([0.87, 1.71, 3.54])#np.array([5.0, 3.0, 1.0])
B = np.array([2.062, 1.276, 1.965]) #np.array([20.0, 36.7, 36.0])
F = np.array([0.001798/3, 0.001653/3, 0.002/3])
#E =  sigmaX_init - np.multiply(F,np.power(r,2)) #np.array([sigmaX_init1-1, sigmaX_init2-1, sigmaX_init3-1])
E = np.array([0.083,0.096,0.129])

sigmaMin = hp.sigma(E,F,rMin)
sigmaMax = hp.sigma(E,F,rMax)
simgaRst = hp.sigma(E,F,rRst)