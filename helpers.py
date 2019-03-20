# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:08:27 2019

This file define helper functions that will be called by main.py
@author: wyue
"""

import numpy as np
from scipy.special import erf
import math

#The design function of clutch
def Y(x1,x2,x3):
    return math.acos((x1+x2)/(x3-x2))

#SigmaY 
def sigmaY(sigma,D):
    VY = np.sum(np.multiply(np.power(D,2),np.power(sigma,2)))
    return np.sqrt(VY)

#Objective function: Unit cost of a product
def U(C,USp,sigmaY,k):
#The parameters are arrays.
    sqrt2 = np.sqrt(2)
    return np.sum(np.divide(C,erf(k/sqrt2)))/erf(USp/sqrt2/sigmaY)
    
    
#Cost-Rate function. The ith component is the ith component in the returned array
def C(A,B,r):
    return np.add(A,np.divide(B,r))
    

#Cost-Rate function. The ith component is the ith component in the returned array
def sigma(E,F,r):
    return np.add(E,np.multiply(F,np.power(r,2)))

#Approximat the deriative of an error function. The error function is approximated
#by a 7th order power series expansion for the error function
def deriv_erf(x):
    return (1.0-x**2+x**4/2.0-x**6/6.0)*2.0/math.sqrt(math.pi)
    
#The derivative of X1 to the clutch design function: dy/dx1. (Also noted as D1)
def dy_dx1(x1,x2,x3):
    return -1.0/(math.sqrt(1-math.pow((x1+x2)/(x3-x2),2))*(x3-x2))
    
#The derivative of X2 to the clutch design function: dy/dx2 (Also noted as D2)
def dy_dx2(x1,x2,x3):
    return -(x1+x3)/(math.sqrt(1.0-math.pow((x1+x2)/(x3-x2),2))*math.pow(x3-x2,2))

#The derivative of X3 to the clutch design function: dy/dx3 (Also noted as D3)
def dy_dx3(x1,x2,x3):
    return (x1+x2)/(math.sqrt(1.0-math.pow((x1+x2)/(x3-x2),2))*math.pow(x3-x2,2))

#dsigma/dr
def dsigmai_dri(Fi,ri):
    return 2*Fi*ri

#dci_dri
def dCi_dri(Bi,ri):
    return -Bi*math.pow(ri,-2)
    
#dsigmaY/dri
def dsigmaY_dri(D,sigma,r,i,dsigma_dr):
    sum = np.sum(np.multiply(np.power(D,2),np.power(sigma,2)))
    return np.power(sum,-0.5)*(D[i]**2)*sigma[i]*dsigma_dr

#dU_dri
def dU_dri(USp,derf_dx,sigmaY,dsigmaY_dri,C,k,dCi_dri,i):
    tem1 = USp/(np.sqrt(2)*sigmaY)
    tem2 = tem1/sigmaY
    tem3 = math.pow(tem1,-2)*deriv_erf(tem1)*tem2*dsigmaY_dri
    tem4 = np.sum(np.divide(C,erf(k/np.sqrt(2))))
    tem5 = dCi_dri/(erf(tem1)*erf(k[i]/np.sqrt(2)))
    return tem3*tem4 + tem5
 
#dU_dki
def dU_dki(USp,sigmaY,ki,Ci):
    sqrt2 = np.sqrt(2)
    tem1 = USp/(np.sqrt(2)*sigmaY)
    return -Ci/sqrt2*deriv_erf(ki/sqrt2)/erf(tem1)/(erf(ki/sqrt2)**2)
    
#Define some helper functions
def produce_satisfactory_output(miu, sigma, Q, TOL):
    N = Q
    LS = miu - TOL
    US= miu + TOL
    components = np.random.normal(miu, sigma, Q)
    #Delete satisfactory components until there are Q satisfactory components
    components = components[np.logical_and(components>LS,components<US)]
    length = components.size
    unsatisfactoryComponents = Q - length
    while unsatisfactoryComponents>0:
        N+=unsatisfactoryComponents
        addedcomponents = np.random.normal(miu, sigma, unsatisfactoryComponents)
        addedcomponents = addedcomponents[np.logical_and(addedcomponents>LS,addedcomponents<US)]
        #Add the new satisfactory components into componnets
        components = np.concatenate((components, addedcomponents))
        length = components.size
        unsatisfactoryComponents = Q - length
    return (components,N)  
 
    
def assembly(X):
    X1=X[0,:]
    X2=X[1,:]
    X3=X[2,:]
    #Y=np.arccos(np.divide((X1+X2),(X3-X2)))
    Y=np.arccos((X1+X2)/(X3-X2))
    return Y


def test(str):
    print(str,test2())
    
def test2():
    return 'test2'






















        