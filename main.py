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
from pandas import DataFrame

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


lamada = 1#0.876
  
NOINSPECT = 1
INSPECTFIXK = 2
INSPECT = 3


scenario =  INSPECT

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
#r = np.array([5.0,5.0,5.0])
#k = np.array([3.0,3.0,3.0])
r = 10 * np.random.rand(3)
#k = 0.2 * np.random.rand(3)
k = np.array([3.0,3.0,3.0])
x = np.concatenate((r,k),axis=0)



grad_r = np.zeros(m)
grad_k = np.zeros(m)


    
def obj_nlopt_inspect(x, grad):
    #retrieve r and k
    num_m = int(x.size/2)
    r = x[0:num_m]
    k = x[num_m:] 
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k)
    #Update Lambda by simulation
    global lamada
    #lamada = hp.updateLambda(D,sigmaX,k,miu,NSample)    
    #sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
     
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v,Sp,Sc) 
        dsigmaY_dki = hp.dsigmaY_dki(D,sigmaX,r,i,k)
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k,i,C,Sc,dsigmaY_dki,Sp) 
    grad_combine = np.concatenate((grad_r,grad_k),axis=0)
    
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    #print(U)
    #print(lamada)
    return U

def obj_nlopt_inspect_fixk(x,grad):
    r = x[0:m]
    k = 3.0*np.ones_like(r)
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k)
    #Update Lambda by simulation
    global lamada
    #lamada = hp.updateLambda(D,sigmaX,k,miu,NSample)    
    #sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
      
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k) 
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)        
    
    if grad.size > 0:
        grad[:] = grad_r #Make sure to assign value using [:]
    #print(U)
    #print(lamada)    
        
    return U

def obj_nlopt_noinspect(x,grad):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    r = x[0:m]   
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k)  
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_noscrap(C,USY,miuY,sigmaY_Taylor,Sp)
    
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)
        grad_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp)
    if grad.size > 0:
        grad[:] = grad_r    #Make sure to assign value using [:]  
    #print(U)
    return U    

    
def optimize(prnt):
    #Unit cost of initial values
    #sigmaX = hp.sigma(E,F,r)    
    #sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #cost = hp.C(A,B,r)
    result = {}    
    if scenario == INSPECT: #Scrap
        opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
        lbK = 0.0
        #ubK = 10.0
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,lbK,lbK,lbK])
        #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
        opt.set_min_objective(obj_nlopt_inspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.concatenate((r,k),axis = 0)
        x = opt.optimize(x0)
        #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
        minf = opt.last_optimum_value()
        result['U'] = minf
        result['r'] = x[0:m]
        result['k'] = x[m:]        
        if prnt==True:              
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
        result['U'] = minf
        result['r'] = x[0:m]
        result['k'] = 3*np.ones((1,m)) 
        if prnt == True:
            print("optimum at ", x[0], x[1],x[2])
            print("minimum value = ", minf)
            print("result code = ", opt.last_optimize_result())             
    
    elif scenario == INSPECTFIXK:
        opt = nlopt.opt(nlopt.LD_MMA, m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue])
        #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
        opt.set_min_objective(obj_nlopt_inspect_fixk)
        opt.set_xtol_rel(1e-4)
        x0 = np.copy(r)
        x = opt.optimize(x0)
        #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
        minf = opt.last_optimum_value()
        result['U'] = minf
        result['r'] = x[0:m]       
        if prnt==True:              
            print("optimum at ", x[0], x[1],x[2])
            print("minimum value = ", minf)
            print("result code = ", opt.last_optimize_result())        
    return result

def oldmethod():
    #Compare this method and CIRP method.       
    if scenario == INSPECT: #Scrap
        #ropt = x[0:m]
        #kopt = x[m:]
        #sigmaopt = hp.sigma(E,F,ropt)
        sigmacompare = np.array([0.09,0.06,0.1])
        sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D,scenario,k)
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
        sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D,scenario,k)
        rcompare = hp.sigmator(sigmacompare,E,F)
        costcompare = hp.C(A,B,rcompare)
        U_compare = hp.U_noscrap(costcompare,USY,miuY,sigmaY_Taylorcompare,Sp)
        print('Old Method minimum value = ', U_compare )


##Compare two scenarios, Section 4.3.3
def comparetwoscenarios(ratio):
    #Scrap costs of components
    global Sc
    Sc = 0*A
    global E  
    #ratio = np.linspace(0.5,4,16)
    
    noinspect_U = np.zeros(len(ratio))
    inspect_U = np.zeros(len(ratio))
    global scenario 
    global r,k,x,grad_r,grad_k
    index = 0
    for rate in ratio:        
        #E = np.array([0.0395,0.0285,0.0495]) #t/6  
        E = np.array([0.036,0.0432,0.054]) 
        E = rate*E 
        scenario = NOINSPECT
        #Average the results
        U = 0
        aveN = 1
        for i in range(aveN):
            result = optimize(False)
            U += result['U']
        U = U/aveN
        noinspect_U[index] = U
        
        scenario = INSPECT 
        U = 0
        for i in range(aveN):
            result = optimize(False)
            U += result['U']
        U = U/aveN
        inspect_U[index] = U
        index +=1
    return [noinspect_U, inspect_U]

def writetofile_U(resultlist):
    df = DataFrame({'noinspect_U':resultlist[0],'inspect_U':resultlist[1]})
    df.to_excel('U.xlsx', sheet_name='sheet1', index=False)
    
    


def computeerror(scenario,ratio):
    global E
    error = np.zeros(len(ratio))
    for idx_rate, rate in enumerate(ratio):
        E = np.array([0.036,0.0432,0.054]) 
        E = rate*E
        result = optimize(False)
        U_equation = result['U']
        r_opt = result['r']
        k_opt = result['k']
        if scenario == INSPECT: 
            U_simulation = hp.U_inspect_simulation(NSample,r_opt,A,B,E,F,k_opt,miu,USY,miuY,Sp,Sc)
            
        elif scenario == NOINSPECT:
             U_simulation = hp.U_noinspect_simulation(NSample,r,A,B,E,F,miu,USY,miuY,Sp,Sc)
             
        error[idx_rate] = (U_equation-U_simulation)/U_simulation
    return error

def comparesatisfactionrate(ratio):
    global E, scenario
    scenario = INSPECT
    gammas = np.zeros((len(ratio),m))
    betas = np.zeros((len(ratio),1))
    ks = np.zeros((len(ratio),m))
    for idx_rate, rate in enumerate(ratio):   
        E = np.array([0.036,0.0432,0.054]) 
        E = rate*E
        result = optimize(False)
        ropt = result['r']
        kopt = result['k']
        gamma = hp.satisfactionrate_component(kopt)
        gammas[idx_rate,:] = gamma
        ks[idx_rate,:] = kopt
        Tol = USY - miuY
        betas[idx_rate] = hp.satisfactionrate_product(Tol,ropt,E,F,D,scenario,kopt)
        
    hp.plotsatisfactoryrate(gammas,betas,ratio,ks)
    

    


##This Method
result = optimize(True)
U_equation = result['U']
r_opt = result['r']
if scenario==INSPECT:
    k_opt = result['k']
else:
    k_opt = 3*np.ones_like(r_opt)
sigma_opt = hp.sigma(E,F,r_opt)
[N,M]=hp.estimateNandM(miu,E,F,r_opt,k_opt,NSample,USY,miuY,scenario)
U_simulation = hp.U_inspect_simulation(NSample,r_opt,A,B,E,F,k_opt,miu,USY,miuY,Sp,Sc)
print('U Equation: ', U_equation)
print('U Simulation: ', U_simulation)
beta = hp.satisfactionrate_product(USY - miuY,r_opt,E,F,D,scenario,k_opt)
print('beta: ', beta)
print('sigmaY: ', hp.sigmaY(sigma_opt,D,scenario,k_opt))
##CIRP Method
#rp = np.array([5.20,3.60,3.40])
#kp = np.array([2.47,2.34,2.93])
#[Np,Mp]=hp.estimateNandM(miu,E,F,rp,kp,NSample,USY,miuY,1)
#Up = hp.U_inspect_simulation(NSample,rp,A,B,E,F,kp,miu,USY,miuY,Sp,Sc)   
##Compare k=3
#kfix=np.array([3,3,3])
#[Nfix,Mfix]=hp.estimateNandM(miu,E,F,r_opt,kfix,NSample,USY,miuY,1)
#Ufix = hp.U_inspect_simulation(NSample,r_opt,A,B,E,F,kfix,miu,USY,miuY,Sp,Sc)


 
#ratio = np.linspace(0.5,4,16)
#resultlist = comparetwoscenarios(ratio)   
#writetofile_U(resultlist)    



#U_simulation = hp.U_scrap_simulation(NSample,r_opt,A,B,E,F,k_opt,miu,USY,miuY,Sp,Sc)
#print('U simulation: ',U_simulation)
#print('error: ', (U_equation-U_simulation)/U_simulation)
#scenario = INSPECT
#ratio = np.linspace(0.5,6,12)
#ratio = np.array([1])
#error = computeerror(scenario,ratio)
#print(error)


#comparesatisfactionrate(ratio)





