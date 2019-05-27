# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:22:10 2019

@author: wyue
"""

#import packages
import numpy as np
#import helper functions
import helpers as hp


#number of components
m = 3
nsample = 10000

miuX1 = 55.291
miuX2 = 22.86
miuX3 = 101.6


sigmaX = np.array([0.11, 0.1, 0.15])
Kcompare = np.array([2.05, 1.385, 3.478]) 
#sigmaX = np.array([1.117320573685910284e-01,1.044633649141074733e-01,1.526516278137779736e-01])
#k = np.array([6.341588383548683, 6.412625456198882, 7.431651149276303])
tol = np.multiply(sigmaX,Kcompare)
#kcompare = np.array([3, 3, 3]) 

D1 = hp.dy_dx1(miuX1,miuX2,miuX3)
D2 = hp.dy_dx2(miuX1,miuX2,miuX3)
D3 = hp.dy_dx3(miuX1,miuX2,miuX3)

D = np.array([D1,D2,D3])

sigmaY_equation = hp.sigmaY(sigmaX,D)

  


#Sigma estimated by simulation - with scrap
(X1_satis,N1) = hp.produce_satisfactory_output(miuX1, sigmaX[0], nsample, tol[0])
(X2_satis,N2) = hp.produce_satisfactory_output(miuX2, sigmaX[1], nsample, tol[1])
(X3_satis,N3) = hp.produce_satisfactory_output(miuX3, sigmaX[2], nsample, tol[2])
X = np.array([X1_satis,X2_satis,X3_satis])
products_simulation_satis = hp.assembly(X)
sigmaY_simulation_satis = np.std(products_simulation_satis)


lambdav =  sigmaY_simulation_satis/sigmaY_equation
print('lambda=',lambdav)