# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:48:35 2019

@author: wyue
Optimizing production rate. Clutch scenario Study
"""
# Check Github

#import packages
import numpy as np
import helpers as hp
import commonfunc as cf
from scipy.optimize import minimize
from scipy.optimize import Bounds
import nlopt
from scipy.spatial import distance
#number of components
m = 3
#Sample size in simulation
NSample = 100000
#Computation Precision
epsilon = 1e-7

smallvalue = 1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 100


miu= np.array([55.291, 22.86, 101.6])

#r = np.array([5, 10.0, 5.0])


#lamada = 1#0.876
   
INSPECT = 1
NOINSPECT = 2
scenario =  INSPECT

p = 0.8
##Tolerance from Choi
CASE = 1
##Tolerance from Zahara
#CASE = 2


(sigmaX_init1,sigmaX_init2, sigmaX_init3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)


sigmaX_init = np.array([sigmaX_init1, sigmaX_init2, sigmaX_init3])

#Estimate cost
#costEst = cf.cost(sigmaX_init)

TX = np.array([TX1,TX2,TX3])
k = np.divide(TX,sigmaX_init)

A = np.array([0.87, 1.71, 3.54])#np.array([5.0, 3.0, 1.0])
B = np.array([2.062, 1.276, 1.965]) #np.array([20.0, 36.7, 36.0])
F = np.array([0.001798/3, 0.001653/3, 0.002/3])
#E =  sigmaX_init - np.multiply(F,np.power(r,2)) #np.array([sigmaX_init1-1, sigmaX_init2-1, sigmaX_init3-1])
E = np.array([0.083,0.096,0.129])
#E = np.array([0,0,0])

#Scrap cost of a product
Sp = np.sum(A)/10
#Scrap costs of components
Sc = A/10

r = hp.sigmator(sigmaX_init,E,F)
#r = np.array([3,5,10])

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])


#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + 0.035


  
#Concatenate r and k into a numpy array
x = np.concatenate((r,k),axis=0)

#obj_scipy_inspect function
def obj_scipy_inspect(x):
    #retrieve r and k
    num_m = int(x.size/2)
    r = x[0:num_m]
    k = x[num_m:] 
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #Update Lambda by simulation
    #lamada = hp.updateLambda(D,sigmaX,k,miu,NSample)
    sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    print(U)
    return U

grad_r = np.zeros(m)
grad_k = np.zeros(m)
#Derivative of object function
def obj_grad_scipy_inspect(x):
    #retrieve r and k
    grad = np.zeros_like(x)
    r = x[0:m]
    k = x[m:] 
    sigmaX = hp.sigma(E,F,r)  
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #Update Lambda by simulation
    #lamada = hp.updateLambda(D,sigmaX,k,miu,NSample)    
    sigmaY_Taylor = lamada*sigmaY_Taylor
    #Compute Unit Cost    
    C = hp.C(A,B,r)       
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)
        
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k[i],C[i],Sc[i])
    grad_combine = np.concatenate((grad_r,grad_k),axis=0)
    grad[:] = grad_combine
    return grad

def obj_scipy_noinspect(x):
    #retrieve r and k
    num_m = int(x.size/2)
    r = x[0:num_m]
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)  
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_noscrap(C,USY,miuY,sigmaY_Taylor,Sp)
    print(U)
    return U

def obj_grad_scipy_noinspect(x):
    #retrieve r and k
    grad = np.zeros(m)
    r = x[0:m]
    k = x[m:] 
    sigmaX = hp.sigma(E,F,r)  
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #Compute Unit Cost    
    C = hp.C(A,B,r)       
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp)
    grad[:] = grad_r    #Make sure to assign value using [:]
    return grad    
    
def obj_nlopt_inspect(x, grad):
    #retrieve r and k
    num_m = int(x.size/2)
    r = x[0:num_m]
    k = x[num_m:] 
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #Update Lambda by simulation
    #global lamada
    lamada = hp.updateLambda(D,sigmaX,k,miu,NSample)    
    sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)

    #Compute Unit Cost    
    C = hp.C(A,B,r)       
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)
        
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k[i],C[i],Sc[i])
    grad_combine = np.concatenate((grad_r,grad_k),axis=0)
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    print(U)
    return U

def obj_nlopt_noinspect(x,grad):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    r = x[0:m]   
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)  
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_noscrap(C,USY,miuY,sigmaY_Taylor,Sp)

    sigmaX = hp.sigma(E,F,r)  
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)     
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp)
    if grad.size > 0:
        grad[:] = grad_r    #Make sure to assign value using [:]  
    print(U)
    return U    

    
ite = 1

def output(x):
    #Retrieve r and k
    global ite
    r = x[0:m]
    #k = x[m:] 
    cost = hp.C(A,B,r)
    for i in range(m):
        print(ite, 'r'+str(i+1)+' ',r[i], 'k'+str(i+1)+' ', k[i], end='')
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    U = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    print(ite, ' U=', U)
    ite +=1


#Unit cost of initial values
sigmaX = hp.sigma(E,F,r)    
sigmaY_Taylor = hp.sigmaY(sigmaX,D)
cost = hp.C(A,B,r)

SCIPY = 0
NLOPT = 1
opt_lib = NLOPT





if opt_lib == SCIPY:  
    if scenario == INSPECT: #Scrap
        #Define Upper and Lower boundaries
        #The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])        
        mbounds = Bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue,largevalue,largevalue,largevalue])        
        U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
        x = np.concatenate((r,k),axis = 0)
        res = minimize(obj_scipy_inspect, x, method='SLSQP', jac=obj_grad_scipy_inspect, #Nelder-Mead #SLSQP
                   options={'ftol': 1e-9, 'maxiter':1000,'disp': True},bounds=mbounds) #constraints=ineq_cons, #,callback=output
    elif scenario == NOINSPECT:
        #Define Upper and Lower boundaries
        #The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])           
        mbounds = Bounds([smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue])          
        U_init = hp.U_noscrap(cost,USY,miuY,sigmaY_Taylor,Sp)
        x = np.copy(r)
        res = minimize(obj_scipy_noinspect, x, method='SLSQP', jac=obj_grad_scipy_noinspect, #Nelder-Mead #SLSQP
               options={'ftol': 1e-9, 'maxiter':1000,'disp': True},bounds=mbounds) #constraints=ineq_cons, #,callback=output
elif opt_lib == NLOPT:
    if scenario == INSPECT: #Scrap
        opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue])
        opt.set_upper_bounds([15,15,15,8,8,8])
        opt.set_min_objective(obj_nlopt_inspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.concatenate((r,k),axis = 0)
        x = opt.optimize(x0)
        U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1],x[2],x[3],x[4],x[5])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())        
    elif scenario == NOINSPECT:
        opt = nlopt.opt(nlopt.LD_MMA, m)
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue])
        opt.set_min_objective(obj_nlopt_noinspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.copy(r)
        x = opt.optimize(x0)
        U_init = hp.U_noscrap(cost,USY,miuY,sigmaY_Taylor,Sp)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1],x[2])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())         



#Compare this method and CIRP method. 
        
if scenario == INSPECT: #Scrap
    ropt = x[0:m]
    kopt = x[m:]
    sigmaopt = hp.sigma(E,F,ropt)
    sigmacompare = np.array([0.11,0.1,0.15])
    sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
    rcompare = hp.sigmator(sigmacompare,E,F)
    costcompare = hp.C(A,B,rcompare)
    kcompare = np.array([2.05, 1.385, 3.478])
    #Update Lambda by simulation
    lamada = hp.updateLambda(D,sigmacompare,kcompare,miu,NSample)   
    #lamada = 0.876
    sigmaY_Taylorcompare = lamada*sigmaY_Taylorcompare    
    U_compare = hp.U_scrap(costcompare,USY,miuY,sigmaY_Taylorcompare,kcompare,Sp,Sc)   
    print('Old Method minimum value = ', U_compare )
elif scenario == NOINSPECT:
    ropt = x
    sigmaopt = hp.sigma(E,F,ropt)
    sigmacompare = np.array([0.11,0.1,0.15])
    sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
    #sigmaY_Taylorcompare = lamada*sigmaY_Taylorcompare
    rcompare = hp.sigmator(sigmacompare,E,F)
    costcompare = hp.C(A,B,rcompare)
    U_compare = hp.U_noscrap(costcompare,USY,miuY,sigmaY_Taylorcompare,Sp)
    print('Old Method minimum value = ', U_compare )





