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


sigmaX = np.array([0.0933, 0.09, 0.11])
kcompare = np.array([6.626964, 1.738889, 6.918]) 
tol = np.multiply(sigmaX,kcompare)
#kcompare = np.array([3, 3, 3]) 

D1 = hp.dy_dx1(miuX1,miuX2,miuX3)
D2 = hp.dy_dx2(miuX1,miuX2,miuX3)
D3 = hp.dy_dx3(miuX1,miuX2,miuX3)

D = np.array([D1,D2,D3])

sigmaY_equation = hp.sigmaY(sigmaX,D)

  
SCRAP = 1
NOSCRAP = 2
scenario =  SCRAP

if scenario == NOSCRAP: 
    X1 = np.random.normal(miuX1, sigmaX[0], nsample)
    X2 = np.random.normal(miuX2, sigmaX[1], nsample)
    X3 = np.random.normal(miuX3, sigmaX[2], nsample)
    X = np.array([X1,X2,X3])
elif scenario ==SCRAP: 
    #Sigma estimated by simulation - with scrap
    (X1_satis,N1) = hp.produce_satisfactory_output(miuX1, sigmaX[0], nsample, tol[0])
    (X2_satis,N2) = hp.produce_satisfactory_output(miuX2, sigmaX[1], nsample, tol[1])
    (X3_satis,N3) = hp.produce_satisfactory_output(miuX3, sigmaX[2], nsample, tol[2])
    X = np.array([X1_satis,X2_satis,X3_satis])
    products_simulation_satis = hp.assembly(X)
    sigmaY_simulation_satis = np.std(products_simulation_satis)


lambdav =  sigmaY_simulation_satis/sigmaY_equation
print('lambda=',lambdav)