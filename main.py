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
NSample = 10000
#Computation Precision
epsilon = 1e-7

smallvalue = 1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 20


miu= np.array([55.291, 22.86, 101.6])

#r = np.array([5, 10.0, 5.0])


#lamada = 1#0.876
   
INSPECT = 1
NOINSPECT = 2
scenario =  NOINSPECT

p = 0.8
##Tolerance from Choi
CASE = 1
##Tolerance from Zahara
#CASE = 2


(sigmaX_init1,sigmaX_init2, sigmaX_init3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)


sigmaX_init = np.array([sigmaX_init1, sigmaX_init2, sigmaX_init3])



A = np.array([0.87, 1.71, 3.54])#np.array([5.0, 3.0, 1.0])
B = np.array([2.062, 1.276, 1.965]) #np.array([20.0, 36.7, 36.0])
F = np.array([0.0020, 0.0013, 0.0040])
E = np.array([0.036,0.0432,0.054])


#Scrap cost of a product
Sp = np.sum(A)/10
#Scrap costs of components
Sc = A/10



D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])


#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + 0.035


  
#Concatenate r and k into a numpy array
r = np.array([5,5,5])
k = np.array([3,3,3])
x = np.concatenate((r,k),axis=0)



grad_r = np.zeros(m)
grad_k = np.zeros(m)


    
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
    #print(U)
    #print(lamada)
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
    #print(U)
    return U    

    
def optimize():
    #Unit cost of initial values
    #sigmaX = hp.sigma(E,F,r)    
    #sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #cost = hp.C(A,B,r)
    
    if scenario == INSPECT: #Scrap
        opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA
        lbK = 2.0
        #ubK = 10.0
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,lbK,lbK,lbK])
        #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
        opt.set_min_objective(obj_nlopt_inspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.concatenate((r,k),axis = 0)
        x = opt.optimize(x0)
        #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
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
        #U_init = hp.U_noscrap(cost,USY,miuY,sigmaY_Taylor,Sp)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1],x[2])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())         


def oldmethod():
    #Compare this method and CIRP method.       
    if scenario == INSPECT: #Scrap
        #ropt = x[0:m]
        #kopt = x[m:]
        #sigmaopt = hp.sigma(E,F,ropt)
        sigmacompare = np.array([0.09,0.06,0.1])
        sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
        rcompare = hp.sigmator(sigmacompare,E,F)
        costcompare = hp.C(A,B,rcompare)
        kcompare = np.array([2.47, 2.34, 2.83])#np.array([2.450709, 1.9927, 3.1678])
        #Update Lambda by simulation
        lamada = hp.updateLambda(D,sigmacompare,kcompare,miu,NSample)   
        #lamada = 0.876
        sigmaY_Taylorcompare = lamada*sigmaY_Taylorcompare    
        U_compare = hp.U_scrap(costcompare,USY,miuY,sigmaY_Taylorcompare,kcompare,Sp,Sc)   
        print('Old Method minimum value = ', U_compare )
    elif scenario == NOINSPECT:
        #ropt = x
        #sigmaopt = hp.sigma(E,F,ropt)
        sigmacompare = np.array([0.09,0.06,0.1])
        sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
        #sigmaY_Taylorcompare = lamada*sigmaY_Taylorcompare
        rcompare = hp.sigmator(sigmacompare,E,F)
        costcompare = hp.C(A,B,rcompare)
        U_compare = hp.U_noscrap(costcompare,USY,miuY,sigmaY_Taylorcompare,Sp)
        print('Old Method minimum value = ', U_compare )


##Compare two scenarios, Section 4.3.3
def comparetwoscenarios():
    #Scrap costs of components
    global Sc
    Sc = 0*A
    global E  
    ratio = np.linspace(1,6,30)
    global scenario 
    global r,k,x,grad_r,grad_k
    for rate in ratio:        
        E = np.array([0.045,0.03,0.05])       
        E = rate*E 
        
        print(rate)
        print("NO Inspect")
        scenario = NOINSPECT

        optimize()
        print("Inspect")
        scenario = INSPECT    
        optimize()
        print("--------------")
    
    
    
##Compute average unit cost. Section 4.3.1 and Section 4.3.2
E = np.array([0.045,0.03,0.05])
E = E*1.8
#optimize()
#oldmethod()    
comparetwoscenarios()   
    
    







