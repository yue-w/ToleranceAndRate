# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:10:04 2019

@author: wyue
"""

#import packages
import numpy as np
import helpers as hp
import commonfunc as cf
import nlopt

#number of components
m = 3

epsilon = 1e-7

#Nominal value of Y
miuY = np.radians(7.0124)
miu= np.array([55.291, 22.86, 101.6])

r = np.array([2.0, 3.0, 2.0])


lamada = 0.87794
   
SCRAP = 1
NOSCRAP = 2
CASE =  SCRAP

p = 0.95
(sigmaX_init1,sigmaX_init2, sigmaX_init3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)

sigmaX_init = np.array([sigmaX_init1, sigmaX_init2, sigmaX_init3])
TX = np.array([TX1,TX2,TX3])
k = np.divide(TX,sigmaX_init)

A = np.array([1.0, 2.0, 3.0])
B = np.array([1.0, 2.0, 3.0])
E = np.array([sigmaX_init1-1, sigmaX_init2-1, sigmaX_init3-1])
F = np.array([0.1, 1.0, 0.5])

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])


#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + np.radians(1.0)


  
#Concatenate r and k into a numpy array
x = np.concatenate((r,k),axis=0)

