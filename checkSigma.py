# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:05:42 2019
Check whether the sigma of Y is corretly estimated by the 
sigma of individual component.
Compared the simulation results to the numerical estimation.
@author: wyue
"""

#import packages
import numpy as np

#import helper functions
import helpers as hp
from scipy.stats import norm 

#number of components
m = 3
#Upper specification limit
USp = np.radians(1.0)
#Nominal value of Y
miuY = np.radians(7.0124)

nsample = 10000

X1 = 55.291
X2 = 22.86
X3 = 101.6


CASE = 2

if CASE == 1:
    # # # #Tolerance from Choi
    TX1 = 0.1798
    TX2 = 0.1653
    TX3 = 0.2  
elif CASE == 2:
    # #Tolerance from Zahara
    TX1 = 0.107696
    TX2 = 0.0127
    TX3 = 0.068072        

p = 0.95
p_alpha = 1-(1-p)/2
z = norm.ppf(p_alpha)

sigmax1 = TX1/z
sigmax2 = TX2/z
sigmax3 = TX3/z

miuX = np.array([X1,X2,X3])
#Nominal value of Y
miuY = np.radians(7.0124)
#specification limits
USY = miuY + np.radians(1.0)
LSY = miuY - np.radians(1.0)


D1 = hp.dy_dx1(miuX[0],miuX[1],miuX[2])
D2 = hp.dy_dx2(miuX[0],miuX[1],miuX[2])
D3 = hp.dy_dx3(miuX[0],miuX[1],miuX[2])

D = np.array([D1,D2,D3])



sigma = np.array([sigmax1,sigmax2,sigmax3])  
sigmaY = hp.sigmaY(sigma,D)

# =============================================================================
# TY = np.radians(1.0)
# zYup = TY/sigmaY
# zYlow = -zYup
# pY_computation = norm.cdf(zYup)-norm.cdf(zYlow)
# =============================================================================

#Prudocts from computation - without scrap
#products_computation =  np.random.normal(miuY, sigmaY, nsample)
#satisfactory_products_computation = products_computation[np.logical_and(products_computation>LSY,products_computation<USY)]

#Sigma estimated by simulation
X1 = np.random.normal(miuX[0], sigmax1, nsample)
X2 = np.random.normal(miuX[1], sigmax2, nsample)
X3 = np.random.normal(miuX[2], sigmax3, nsample)
X = np.array([X1,X2,X3])
products_simulation = hp.assembly(X)
sigmaY_simulation = np.std(products_simulation)

#Sigma estimated by simulation - with scrap
(X1_satis,N1) = hp.produce_satisfactory_output(miuX[0], sigmax1, nsample, TX1)
(X2_satis,N2) = hp.produce_satisfactory_output(miuX[1], sigmax2, nsample, TX2)
(X3_satis,N3) = hp.produce_satisfactory_output(miuX[2], sigmax3, nsample, TX3)
X_satis = np.array([X1_satis,X2_satis,X3_satis])
products_simulation_satis = hp.assembly(X_satis)
sigmaY_simulation_satis = np.std(products_simulation_satis)

errorTS = (sigmaY - sigmaY_simulation)/sigmaY_simulation * 100
errorTS_satis = (sigmaY - sigmaY_simulation_satis)/sigmaY_simulation_satis * 100
print('sigmaY estimated from simulation (no scrap) = ' + str(sigmaY_simulation))
print('sigmaY computed from Taylor expansion (no scrap) = ' + str(sigmaY))
print('Error in Taylor Series = ' + str(errorTS)+'%')
print('sigmaY estimated from simulation (with scrap) = ' + str(sigmaY_simulation_satis)) 
print('Error in Taylor Series with scrap = ' + str(errorTS_satis)+'%')   



