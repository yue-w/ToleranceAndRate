# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:08:27 2019

This file define helper functions that will be called by main.py
@author: wyue
"""

import numpy as np
from scipy.special import erf
import math
from scipy.special import factorial

#n is the order of polimonial function to approximate the error function and its derivative
n = 70
#The design function of clutch
def Y(x1,x2,x3):
    return math.acos((x1+x2)/(x3-x2))

#SigmaY 
def sigmaY(sigmaX,D):
    
    VY = np.sum(np.multiply(np.power(D,2),np.power(sigmaX,2)))
    return np.sqrt(VY)

#Objective function: Unit cost of a product
def U_scrap(C,USY,miuY,sigmaYp,k,Sp,Sc):
 #The parameters are arrays.   
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    erfV = erf(USp/sqrt2/sigmaYp)
    erfK = erf(k/sqrt2)
    term1 = np.sum(np.divide(C,erfK))/erfV
    tem = np.divide(1,erfK)
    tem2 = np.subtract(tem,1)
    tem3 = np.multiply(Sc,tem2)
    term2 = np.sum(tem3)/erfV
    term3 = (1/erfV - 1) * Sp
    return term1 + term2 + term3
    #return np.sum(np.divide(C,erf(k/sqrt2)))/erf(USp/sqrt2/sigmaYp)

#Objective function: Unit cost of a product
def U_noscrap(C,USY,miuY,sigmaY,Sp):
 #The parameters are arrays.   
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    erfV = erf(USp/sqrt2/sigmaY) 
    term1 = np.sum(C)/erfV
    term2 = (1/erfV-1) * Sp
    
    return term1 + term2
    
#Cost-Rate function. The ith component is the ith component in the returned array
def C(A,B,r):
    return np.add(A,np.divide(B,r))
    

#Cost-Rate function. The ith component is the ith component in the returned array
def sigma(E,F,r):
    return np.add(E,np.multiply(F,np.power(r,2)))

#Approximat the deriative of an error function. The error function is approximated
#by a 7th order power series expansion for the error function
def derf_dx(x,n):
    #return (1.0-x**2+x**4/2.0-x**6/6.0)*2.0/math.sqrt(math.pi)
    val = 0
    token = 1
    for i in range(0,n):
        val += token * np.power(x,2*i) / factorial(i)
        token *= -1 
    val = val * 2 / np.sqrt(np.pi)
    return val
        
    
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

#dU_dri No scrap
def dU_dri_noscrap(USY, miuY,sigmaY,C,k,i,dsigmaY_dri,dCi_dri,Sp):
    USp=USY - miuY
    tem1 = USp/(np.sqrt(2)*sigmaY)
    tem2 = tem1/sigmaY
    tem3 = math.pow(erf(tem1),-2)*derf_dx(tem1,n)*tem2*dsigmaY_dri
    tem4 = np.sum(C) + Sp
    tem5 = dCi_dri/erf(tem1)
    return tem3*tem4 + tem5

#dU_dri Scrap
def dU_dri_scrap(USY, miuY,sigmaYp,C,k,i,lamada,dsigmaY_dri,dCi_dri,Sp,Sc):
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    tem1 = USp/(sqrt2*sigmaYp)
    tem2 = tem1/sigmaYp
    tem3 = math.pow(erf(tem1),-2)*derf_dx(tem1,n)*tem2*dsigmaY_dri*lamada
    tem4 = np.sum(np.multiply((np.divide(1,erf(k/sqrt2))-1),Sc))
    tem5 = np.sum(np.divide(C,erf(k/sqrt2))) + tem4 + Sp
    tem6 = dCi_dri/(erf(tem1)*erf(k[i]/sqrt2))
    return tem3*tem5 + tem6

#dU_dki
def dU_dki_scrap(USY, miuY,sigmaYp,ki,Ci,Sci):
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    tem1 = USp/(sqrt2*sigmaYp)
    tem2 = ki/sqrt2
    return -np.add(Ci,Sci)/sqrt2*derf_dx(tem2,n)/erf(tem1)/np.power(erf(tem2),2)
    
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
    dividV = np.divide((X1+X2),(X3-X2))
    dividV = dividV[np.logical_and(dividV>=-1,dividV<=1)]
    Y=np.arccos(dividV)
    #Y=np.arccos(np.divide((X1+X2),(X3-X2))) 
    return Y

def save_data_csv(filename,data):
    np.savetxt(filename, data, delimiter=' ')

def sigmator(sigma,E,F):
    r = np.sqrt(np.divide(np.subtract(sigma,E),F))
    return r

def updateLambda(D,sigmaX,k,miuX,NSample):
    tol = np.multiply(sigmaX,k)
    sigmaY_equation = sigmaY(sigmaX,D)
    
    #Sigma estimated by simulation - with scrap
    (X1_satis,N1) = produce_satisfactory_output(miuX[0], sigmaX[0], NSample, tol[0])
    (X2_satis,N2) = produce_satisfactory_output(miuX[1], sigmaX[1], NSample, tol[1])
    (X3_satis,N3) = produce_satisfactory_output(miuX[2], sigmaX[2], NSample, tol[2])
    X = np.array([X1_satis,X2_satis,X3_satis])
    products_simulation_satis = assembly(X)
    sigmaY_simulation_satis = np.std(products_simulation_satis)
    
    
    lamada =  sigmaY_simulation_satis/sigmaY_equation    
    
    return lamada

def simulateSigmaY(D,sigmaX,miuX,NSample):
   #Estimate sigmaY by simulation instead of equation. 
    X1 = np.random.normal(miuX[0], sigmaX[0], NSample)
    X2 = np.random.normal(miuX[1], sigmaX[1], NSample)
    X3 = np.random.normal(miuX[2], sigmaX[2], NSample)
    X = np.array([X1,X2,X3])

    products_simulation_satis = assembly(X)
    sigmaY_simulation = np.std(products_simulation_satis)

    return sigmaY_simulation














        