# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:54:11 2019
Check the gradient of ri and ki bomputed by equation and by numerical method
@author: wyue
"""

#import packages
import numpy as np

#import helper functions
import helpers as hp
from scipy.stats import norm 

# =============================================================================
# #Variables to be optimized
# r = np.array([1.0, 1.0, 1.0])
# #We assume that for each component the satisfactory rate is 0.9
# p = 0.95
# z = norm.ppf(p)
# k = np.array([z,z,z])
# =============================================================================


#number of components
m = 3
#Upper specification limit
USp = np.radians(1.0)
#Nominal value of Y
miuY = np.radians(7.0124)

USx1 = 0.02
USx2 = 0.01
USx3 = 0.03

r = np.array([1.0, 1.0, 1.0])

p = 0.9
p_alpha = 1-(1-p)/2
k_val = norm.ppf(p_alpha)
k = np.ones(3)*k_val

sigmax1 = USx1/k[0]
sigmax2 = USx2/k[1]
sigmax3 = USx3/k[2]

A = np.array([1.0, 2.0, 3.0])
B = np.array([1.0, 2.0, 3.0])
E = np.array([sigmax1-1, sigmax2-1, sigmax3-1])
F = np.array([1.0, 1.0, 1.0])

miu= np.array([55.291,22.86,101.6])


epsilon = 1e-7

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])

#Compute Unit Cost of initial value
C = hp.C(A,B,r)
sigma = hp.sigma(E,F,r)  
sigmaY = hp.sigmaY(sigma,D)

U = hp.U(C,USp,sigmaY,k)

dec_estmate = np.zeros(m)
for i in range(0,m):  
    ri_add_epsilon = np.copy(r)
    ri_minus_epsilon = np.copy(r)
    ri_add_epsilon[i] +=  epsilon
    ri_minus_epsilon[i] -=  epsilon
    sigma_plus = hp.sigma(E,F,ri_add_epsilon)  
    sigma_minus = hp.sigma(E,F,ri_minus_epsilon)  
    C_plus = hp.C(A,B,ri_add_epsilon)
    C_minus = hp.C(A,B,ri_minus_epsilon)
    sigmaY_plus = hp.sigmaY(sigma_plus,D)
    sigmaY_minus = hp.sigmaY(sigma_minus,D)
    dec_estmate[i] = (hp.U(C_plus,USp,sigmaY_plus,k) - hp.U(C_minus,USp,sigmaY_minus,k))/(2*epsilon)
    print('r',i,'=',dec_estmate[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    