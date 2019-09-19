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
import matplotlib.pyplot as plt

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


 
NOINSPECT = 1
INSPECTFIXK = 2
INSPECT = 3


scenario =  INSPECTFIXK


####Hub Roller Cage
A = np.array([0.88, 0.42, 1.12]) + 0.1 
B = np.array([2.5, 1.0, 5.0]) #np.array([20.0, 36.7, 36.0])

E = np.array([0.0232*2, 0.0232, 0.0232*3]) * 1.2 
F = np.array([0.0020*2, 0.0020, 0.0020*3.0]) * 0.12

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
r = np.array([5.0,5.0,5.0])
#k = np.array([3.0,3.0,3.0])
#r = 10 * np.random.rand(3)
#k = 0.2 * np.random.rand(3)
k = np.array([3.0,3.0,3.0])
x = np.concatenate((r,k),axis=0)



grad_r = np.zeros(m)
grad_k = np.zeros(m)


    
def obj_nlopt_inspect(x, grad, para):
    #retrieve r and k
    A = para[0]
    B = para[1]
    E = para[2]
    F = para[3]
    num_m = int(x.size/2)
    r = x[0:num_m]
    k = x[num_m:] 
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k)

    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
     
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp,Sc) 
        dsigmaY_dki = hp.dsigmaY_dki(D,sigmaX,r,i,k)
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k,i,C,Sc,dsigmaY_dki,Sp) 
    grad_combine = np.concatenate((grad_r,grad_k),axis=0)
    
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    #print(U)
    #print(lamada)
    return U

def obj_nlopt_inspect_fixk(x,grad,para):
    A = para[0]
    B = para[1]
    E = para[2]
    F = para[3]
    r = x[0:m]
    k = 3.0*np.ones_like(r)
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k) 
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
      
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k) 
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)        
    
    if grad.size > 0:
        grad[:] = grad_r #Make sure to assign value using [:]
    #print(U)
    #print(lamada)    
        
    return U

def obj_nlopt_noinspect(x,grad,para):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    A = para[0]
    B = para[1]
    E = para[2]
    F = para[3]    
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

    
def optimize(prnt,para):
    #Unit cost of initial values
    #sigmaX = hp.sigma(E,F,r)    
    #sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #cost = hp.C(A,B,r)
    result = {}    
    if scenario == INSPECT: #Scrap
        opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
        #ubK = 10.0
        #opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,lbK,lbK,lbK])
        #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
        opt.set_min_objective(lambda x,grad: obj_nlopt_inspect(x,grad,para))
        #opt.set_min_objective(obj_nlopt_inspect)
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
        opt.set_min_objective(lambda x,grad: obj_nlopt_noinspect(x,grad,para))
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
        #opt.set_lower_bounds([smallvalue,smallvalue,smallvalue])
        #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
        opt.set_min_objective(lambda x,grad: obj_nlopt_inspect_fixk(x,grad,para))
        opt.set_xtol_rel(1e-4)
        x0 = np.copy(r)
        x = opt.optimize(x0)
        #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
        minf = opt.last_optimum_value()
        result['U'] = minf
        result['r'] = x[0:m]  
        result['k'] = 3*np.ones((1,m)) 
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
        #lamada = hp.updateLambda(D,sigmacompare,kcompare,miu,NSample)   
        #lamada = 0.876  
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
    #ratio = np.linspace(0.5,4,16)
    
    noinspect_U = np.zeros(len(ratio))
    noinspect_Up = np.zeros(len(ratio))
    inspect_U_fix = np.zeros(len(ratio))
    inspect_Up_fix = np.zeros(len(ratio))
    inspect_U = np.zeros(len(ratio))
    inspect_Up = np.zeros(len(ratio))
    global scenario 
    global r,k,x,grad_r,grad_k
    para = np.array([A,B,E,F])
    for index, rate in enumerate(ratio):    
        para[2] = rate*E 
        scenario = NOINSPECT
        #Average the results
        U = 0
        Up = 0
        aveN = 5
        for i in range(aveN):
            result = optimize(False,para)
            U += result['U']
            ropt = result['r']
            Up += hp.U_noinspect_simulation(NSample,ropt,A,B,para[2],F,miu,USY,miuY,Sp,Sc)
        U = U/aveN
        Up = Up/aveN
        noinspect_U[index] = U
        noinspect_Up[index] = Up
        
        scenario = INSPECTFIXK  
        U = 0
        Up = 0
        for i in range(aveN):
            result = optimize(False,para)
            U += result['U']
            ropt = result['r']
            kopt = 3.0*np.ones_like(ropt)
            Up += hp.U_inspect_simulation(NSample,ropt,A,B,para[2],F,kopt,miu,USY,miuY,Sp,Sc)
        U = U/aveN
        Up = Up/aveN
        inspect_U_fix[index] = U
        inspect_Up_fix[index] = Up
        
        scenario = INSPECT 
        U = 0
        Up = 0
        for i in range(aveN):
            result = optimize(False,para)
            ropt = result['r']
            kopt = result['k']            
            U += result['U']
            Up += hp.U_inspect_simulation(NSample,ropt,A,B,para[2],F,kopt,miu,USY,miuY,Sp,Sc)            
        U = U/aveN
        Up = Up/aveN
        inspect_U[index] = U
        inspect_Up[index] = Up
      
    return [noinspect_U,noinspect_Up, inspect_U_fix,inspect_Up_fix, inspect_U,inspect_Up]

def writetofile_U(resultlist):
    df = DataFrame({'noinspect_U':resultlist[0],'noinspect_Up':resultlist[1],'inspect_U_fixk':resultlist[2],'inspect_Up_fixk':resultlist[3],'inspect_U':resultlist[4],'inspect_Up':resultlist[5]})
    df.to_excel('U.xlsx', sheet_name='sheet1', index=False)
    
    


def computeerror(scenario,ratio):
    error = np.zeros(len(ratio))
    para = np.array([A,B,E,F])
    for idx_rate, rate in enumerate(ratio):
        para[2] = E*rate
        result = optimize(False,para)
        U_equation = result['U']
        r_opt = result['r']
        k_opt = result['k']
        if scenario == INSPECT: 
            U_simulation = hp.U_inspect_simulation(NSample,r_opt,A,B,para[2],F,k_opt,miu,USY,miuY,Sp,Sc)
            
        elif scenario == NOINSPECT:
             U_simulation = hp.U_noinspect_simulation(NSample,r_opt,A,B,para[2],F,miu,USY,miuY,Sp,Sc)
        
        elif scenario == INSPECTFIXK:
            k_opt = 3*np.ones_like(r_opt)
            U_simulation = hp.U_inspect_simulation(NSample,r_opt,A,B,para[2],F,k_opt,miu,USY,miuY,Sp,Sc)
        
        error[idx_rate] = np.abs(U_equation-U_simulation)/U_simulation * 100
    return error

def comparesatisfactionrateandk(ratio,scenario):
    gammas = np.zeros((len(ratio),m))
    betas = np.zeros((len(ratio),1))
    ks = np.zeros((len(ratio),m))
    para = np.array([A,B,E,F])
    for idx_rate, rate in enumerate(ratio):  
        para[2] = E*rate
        result = optimize(False,para)
        ropt = result['r']
        if scenario == 3:
            kopt = result['k']
        else:
            kopt = 3*np.ones(m)
        ks[idx_rate,:] = kopt
        satisfactionrate = hp.satisfactionrate_component_product(miu,para[2],F,ropt,kopt,NSample,USY,miuY,scenario)
        gammas[idx_rate] = satisfactionrate['gammas']
        betas[idx_rate] = satisfactionrate['beta']
    hp.plotsatisfactoryrateandk(gammas,betas,ratio,ks)
    

def plotC(A,B,r):
    fig, ax1 = plt.subplots()
    ax1.scatter(r, hp.C(A[0],B[0],r), s=4, label="Hub", color='r' )
    ax1.scatter(r, hp.C(A[1],B[1],r), s=4, label="Roller", color='g' )
    ax1.scatter(r, hp.C(A[2],B[2],r), s=4, label="Cage", color='b' )
    #ax1.scatter(krange, np.tanh(krange), s=0.1, label="tanh", color='y' )
    ax1.set_ylabel('C') 
    ax1.set_xlabel('r')
    ax1.legend()
    ax1.grid(True)
    plt.show()     
    fig.savefig(fname='cost',dpi=300)
    #fig.savefig('3dPlot.tif')

def plotsigma(E,F,r):
    F = np.array([0.0020, 0.0013, 0.001050]) 
    E = np.array([0.036,0.0432,0.014]) 
    fig, ax1 = plt.subplots()
    ax1.scatter(r, hp.sigma(E[0],F[0],r), s=4, label="Hub", color='r' )
    ax1.scatter(r, hp.sigma(E[1],F[1],r), s=4, label="Roller", color='g' )
    ax1.scatter(r, hp.sigma(E[2],F[2],r), s=4, label="Cage", color='b' )
    #ax1.scatter(krange, np.tanh(krange), s=0.1, label="tanh", color='y' )
    ax1.set_ylabel('Sigma') 
    ax1.set_xlabel('r')
    ax1.legend()
    ax1.grid(True)
    plt.show()     
    fig.savefig(fname='sigma',dpi=300)    

def casestudy_U():
    para = np.array([A,B,E,F])
    result = optimize(True,para)
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
    satisfactionrate = hp.satisfactionrate_component_product(miu,E,F,r,k,NSample,USY,miuY,scenario)
    print('beta: ', satisfactionrate['beta'])
    print('sigmaY: ', hp.sigmaY(sigma_opt,D,scenario,k_opt))
    print('opt cost:', hp.C(A,B,r_opt))
    print('N: ',N)
    print('M: ', M)
    print('Gama', satisfactionrate['gammas'])
    
    
def CIRP():
    sigmap = np.array([0.317,0.1135,0.4606])/3
    rp = hp.sigmator(sigmap,E,F)
    kp = np.array([3,3,3])
    [Np,Mp]=hp.estimateNandM(miu,E,F,rp,kp,NSample,USY,miuY,scenario)
    Up = hp.U_inspect_simulation(NSample,rp,A,B,E,F,kp,miu,USY,miuY,Sp,Sc)   
    print('CIRP U: ', Up)
    print('CIRP N ', Np)
    print('CIRP M ', Mp)


    
#plotC(A,B,np.arange(1,10,0.1))
#plotsigma(E,F,np.arange(1,10,0.1))
    
casestudy_U()
CIRP()

ratio = np.linspace(0.5,2.0,15)
error = computeerror(scenario,ratio)
print(error)
hp.scatterplot(ratio,error,'ratio','error%',save=False,filename='scatterplot') 

#ratio = np.linspace(1,2,10)
#resultlist = comparetwoscenarios(ratio)   
#writetofile_U(resultlist)  
#hp.plotU(ratio)
print('done')

#comparesatisfactionrateandk(ratio,scenario)





