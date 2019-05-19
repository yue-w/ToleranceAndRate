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
#from scipy.stats import norm 
import commonfunc as cf

#number of components
m = 3

nsample = 10000

miuX1 = 55.291
miuX2 = 22.86
miuX3 = 101.6
CASE = 2
p = 0.95
(sigmax1,sigmax2, sigmax3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)


miuX = np.array([miuX1,miuX2,miuX3])



D1 = hp.dy_dx1(miuX[0],miuX[1],miuX[2])
D2 = hp.dy_dx2(miuX[0],miuX[1],miuX[2])
D3 = hp.dy_dx3(miuX[0],miuX[1],miuX[2])

D = np.array([D1,D2,D3])



sigma = np.array([sigmax1,sigmax2,sigmax3])  
sigmaY = hp.sigmaY(sigma,D)


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



