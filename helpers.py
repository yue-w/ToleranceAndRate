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
from scipy.stats import norm 
import matplotlib.pyplot as plt
import pandas as pd


#n is the order of polimonial function to approximate the error function and its derivative
n = 70
THETA = 1/erf(4/erf(2))
#The design function of clutch
def Y(x1,x2,x3):
    
    return math.acos((x1+x2)/(x3-x2))

a = 2.26
c = 1.09
b = 1 + 1/np.exp(a*c)
d = -1/np.exp(a*c)

def h(x,a,b,c,d):
    #return np.divide(2*np.ones_like(x),1+np.exp(-1*a*x)) - 1  
    return np.divide(b*np.ones_like(x),1+np.exp(-a*(x-c)))+d

def dh_dx(x,a,b,c,d):
    #tem = np.exp(-1*a*x)
    #v = np.divide(2*a*tem,np.power(1+tem,2))
    tem = np.exp(-1*a*x+a*c)
    return b*a*np.multiply(np.power(1+tem,-2),tem)
    
  
#SigmaY 
def sigmaY(sigmaX,D,scenario,k):
    if scenario == 1 or scenario ==2:
        VY = np.sum(np.multiply(np.power(D,2),np.power(sigmaX,2)))
        V = np.sqrt(VY)
    elif scenario == 3:
        tem = np.multiply(np.power(D,2),np.power(sigmaX,2))
        tem2 = np.multiply(tem,np.power(h(k,a,b,c,d),2))
        VY = np.sum(tem2)
        V =  np.sqrt(VY)  #1.0/erf(THETA/np.sqrt(2)) * np.sqrt(VY)
    return V

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
def dsigmaY_dri(D,sigma,r,i,dsigma_dr,scenario,k):
    if scenario == 1 or scenario == 2:
        sum = np.sum(np.multiply(np.power(D,2),np.power(sigma,2)))
        v = np.power(sum,-0.5)*(D[i]**2)*sigma[i]*dsigma_dr
        
    if scenario == 3:
        hk = h(k,a,b,c,d)
        tem1 = np.multiply(np.power(D,2),np.power(sigma,2))
        tem2 = np.multiply(tem1, np.power(hk,2))
        tem3 = np.power(np.sum(tem2),-0.5)
        v = tem3*np.power(D[i],2)*np.power(hk[i],2)*sigma[i]*dsigma_dr
        
    return v


def dsigmaY_dki(D,sigma,r,i,k):
    tem1 = np.multiply(np.power(D,2),np.power(sigma,2))
    hk = h(k,a,b,c,d)
    tem2 = np.multiply(tem1, np.power(hk,2))
    tem3 = np.power(np.sum(tem2),-0.5)  
    v = tem3*np.power(D[i],2)*np.power(sigma[i],2)*hk[i]*dh_dx(k[i],a,b,c,d)
    return v 

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
def dU_dri_scrap(USY, miuY,sigmaYp,C,k,i,dsigmaY_dri,dCi_dri,Sp,Sc):
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    tem1 = USp/(sqrt2*sigmaYp)
    tem2 = tem1/sigmaYp
    tem3 = math.pow(erf(tem1),-2)*derf_dx(tem1,n)*tem2*dsigmaY_dri
    tem4 = np.sum(np.multiply((np.divide(1,erf(k/sqrt2))-1),Sc))
    tem5 = np.sum(np.divide(C,erf(k/sqrt2))) + tem4 + Sp
    tem6 = dCi_dri/(erf(tem1)*erf(k[i]/sqrt2))
    return tem3*tem5 + tem6

#dU_dki
def dU_dki_scrap(USY, miuY,sigmaY,k,i,C,Sc,dsigmaY_dki,Sp):
    sqrt2 = np.sqrt(2)
    tem1 = (USY - miuY)/(sqrt2*sigmaY)
    v1 = np.power(erf(tem1),-2)
    
    tem2 = np.divide(C+Sc,erf(k/sqrt2)) - Sc
    tem3 = Sp + np.sum(tem2)
    v2 = tem1/sigmaY*dsigmaY_dki*derf_dx(tem1,n)*tem3
    
    v3 = erf(tem1)*np.power(erf(k[i]/sqrt2),-2)*(Sc[i]+C[i])/sqrt2*derf_dx(k[i]/sqrt2,n)

    
    dudk = v1*(v2 - v3)
    return dudk 
    
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


def simulateSigmaY(D,sigmaX,miuX,NSample):
   #Estimate sigmaY by simulation instead of equation. 
    X1 = np.random.normal(miuX[0], sigmaX[0], NSample)
    X2 = np.random.normal(miuX[1], sigmaX[1], NSample)
    X3 = np.random.normal(miuX[2], sigmaX[2], NSample)
    X = np.array([X1,X2,X3])

    products_simulation_satis = assembly(X)
    sigmaY_simulation = np.std(products_simulation_satis)

    return sigmaY_simulation


def U_inspect_simulation(NSample,r,A,B,E,F,k,miuX,USY,miuY,Sp,Sc):
    sigmaX = sigma(E,F,r)
    tol = np.multiply(sigmaX,k)
    (X1_satis,N1) = produce_satisfactory_output(miuX[0], sigmaX[0], NSample, tol[0])
    (X2_satis,N2) = produce_satisfactory_output(miuX[1], sigmaX[1], NSample, tol[1])
    (X3_satis,N3) = produce_satisfactory_output(miuX[2], sigmaX[2], NSample, tol[2])
    #Number of components processed
    N = np.array([N1,N2,N3])
    #Number of components scrapped
    Np = np.abs(N - NSample)
    #Process cost of components
    Cp = C(A,B,r)
    X = np.array([X1_satis,X2_satis,X3_satis])
    Y = assembly(X)
    Y = Y[np.logical_and(Y>=2*miuY-USY,Y<=USY)]
    #Total cost 
    Ct = np.sum(np.multiply(N,Cp)) + np.sum(np.multiply(Np,Sc)) + Sp*(np.abs(NSample-len(Y)))
    U = Ct/len(Y)
    return U
    
def U_noinspect_simulation(NSample,r,A,B,E,F,miuX,USY,miuY,Sp,Sc):
    sigmaX = sigma(E,F,r)
    #tol = np.multiply(sigmaX,k)    
    X1 = np.random.normal(miuX[0], sigmaX[0], NSample)
    X2 = np.random.normal(miuX[1], sigmaX[1], NSample)
    X3 = np.random.normal(miuX[2], sigmaX[2], NSample)
    #Process cost of components
    Cp = C(A,B,r)
    X = np.array([X1,X2,X3])
    Y = assembly(X)
    Y = Y[np.logical_and(Y>=2*miuY-USY,Y<=USY)]    
    #Total cost 
    Ct = np.sum(NSample*Cp) + Sp*(NSample-len(Y))
    U = Ct/len(Y)
    return U    

def estimateNandM(miuX,E,F,r,k,NSample,USY,miuY,scenario):
    sigmaX = sigma(E,F,r)
    if scenario == 1:
        X1 = np.random.normal(miuX[0], sigmaX[0], NSample)
        X2 = np.random.normal(miuX[1], sigmaX[1], NSample)
        X3 = np.random.normal(miuX[2], sigmaX[2], NSample) 
        N1 = NSample
        N2 = NSample
        N3 = NSample        
    elif scenario == 2 or scenario == 3:
        tol = np.multiply(sigmaX,k)
        (X1,N1) = produce_satisfactory_output(miuX[0], sigmaX[0], NSample, tol[0])
        (X2,N2) = produce_satisfactory_output(miuX[1], sigmaX[1], NSample, tol[1])
        (X3,N3) = produce_satisfactory_output(miuX[2], sigmaX[2], NSample, tol[2]) 
    N = np.array([N1,N2,N3])
    X = np.array([X1,X2,X3])
    Y = assembly(X)
    Y = Y[np.logical_and(Y>=2*miuY-USY,Y<=USY)]    
    M = len(Y)
    return [N,M]

def satisfactionrate_component(Z):
    px = norm.cdf(Z)
    gamma = 1-2*(1-px)    
    return gamma

def satisfactionrate_component_product(miuX,E,F,r,k,NSample,USY,miuY,scenario):
    ##Use simulation to calculate beta     
    [N,M] = estimateNandM(miuX,E,F,r,k,NSample,USY,miuY,scenario)
    gamma = NSample/N*100
    beta = M/NSample*100
    satisfactionrate = {}
    satisfactionrate['gammas'] = gamma
    satisfactionrate['beta'] = beta
    return satisfactionrate


def plotsatisfactoryrateandk(gammas,betas,ratio,kopt):
    fig, ax1 = plt.subplots()
    #plot gamma1
    ax1.scatter(ratio, gammas[:,0], label="gamma1", color='r' )
    #plot gamma2   
    ax1.scatter(ratio, gammas[:,1], label="gamma2", color='g' ) 
    #plot gamma3
    ax1.scatter(ratio, gammas[:,2], label="gamma3", color='b' ) 
    #plot beta
    ax1.scatter(ratio, betas, label="beta", color='c', marker='*' ) 
    ax1.set_ylabel('Satisfaction rate') 
    ax1.set_xlabel('Ratio')
    

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
     
    ax2.set_ylabel(r'k')  # we already handled the x-label with ax1
    ax2.scatter(ratio, kopt[:,0],label="k1", color='m', marker='+' )
    ax2.scatter(ratio, kopt[:,1],label="k2", color='y', marker='+')
    ax2.scatter(ratio, kopt[:,2],label="k3", color='k', marker='+')
    ax2.tick_params(axis='y')
    ax2.legend(loc=1)
       
    ax1.legend(loc=3)
    ax1.grid(True)    
    plt.show()    
    fig.savefig(fname='satisfactionrate',dpi=300)
    fig.savefig('3dPlot.tif')    
    
def scatterplot(x,y,xlabel,ylabel,save=False,filename='scatterplot'):
    fig, ax1 = plt.subplots()
    ax1.scatter(x, y, s=4, color='r' )
    #ax1.scatter(krange, np.tanh(krange), s=0.1, label="tanh", color='y' )
    ax1.set_ylabel(ylabel) 
    ax1.set_xlabel(xlabel)
    #ax1.legend()
    ax1.grid(True)
    plt.show()     
    fig.savefig(fname='cost',dpi=300)    
    
def plotU(xaxis):
    df=pd.read_excel("U.xlsx",sheet_name='sheet1')
    nonins_Up = df['noinspect_U']
    ins_Up = df['inspect_U_fixk']
    inspect_Up = df['inspect_U']
    
    fig, ax1 = plt.subplots()
    ax1.scatter(xaxis, nonins_Up, label="No inspect", color='r' )
    ax1.scatter(xaxis, ins_Up, label="k=3", color='b' )
    ax1.scatter(xaxis, inspect_Up, label="k opt", color='g' )
    ax1.set_ylabel('U') 
    
    ax1.set_xlabel('Ratio')
    
    ax1.legend()
    ax1.grid(True)
    
    plt.show()
    
    fig.savefig(fname='U_Scenarios',dpi=300)
    #fig.savefig('3dPlot.tif')    
    

    
    
        