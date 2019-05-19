# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:48:35 2019

@author: wyue
Optimizing production rate. Clutch scenario Study
"""

#import packages
import numpy as np
import helpers as hp
import commonfunc as cf
from scipy.optimize import minimize
from scipy.optimize import Bounds
import nlopt

#number of components
m = 3

epsilon = 1e-7

smallvalue = 1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 100


miu= np.array([55.291, 22.86, 101.6])

#r = np.array([5, 10.0, 5.0])


lamada = 0.87794
   
SCRAP = 1
NOSCRAP = 2
scenario =  SCRAP

p = 0.9
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
E = np.array([0,0,0])

r = hp.sigmator(sigmaX_init,E,F)

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
    sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k)
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
    sigmaY_Taylor = lamada*sigmaY_Taylor
    #Compute Unit Cost    
    C = hp.C(A,B,r)       
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v)
        
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k[i],C[i])
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
    U = hp.U_noscrap(C,USY,miuY,sigmaY_Taylor)
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
        grad_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v)
    grad[:] = grad_r    #Make sure to assign value using [:]
    return grad    
    
def obj_nlopt_inspect(x, grad):
    #retrieve r and k
    num_m = int(x.size/2)
    r = x[0:num_m]
    k = x[num_m:] 
    sigmaX = hp.sigma(E,F,r)    
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    sigmaY_Taylor = lamada*sigmaY_Taylor    
    #Compute Unit Cost    
    C = hp.C(A,B,r)
    U = hp.U_scrap(C,USY,miuY,sigmaY_Taylor,k)

    sigmaX = hp.sigma(E,F,r)  
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    sigmaY_Taylor = lamada*sigmaY_Taylor
    #Compute Unit Cost    
    C = hp.C(A,B,r)       
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v)
        
        grad_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k[i],C[i])
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
    U = hp.U_noscrap(C,USY,miuY,sigmaY_Taylor)

    sigmaX = hp.sigma(E,F,r)  
    sigmaY_Taylor = hp.sigmaY(sigmaX,D)     
    for i in range(0,m):  # Change this for loop to vectorization 
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v)
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
    U = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k)
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
    if scenario == SCRAP: #Scrap
        #Define Upper and Lower boundaries
        #The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])        
        mbounds = Bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue,largevalue,largevalue,largevalue])        
        U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k)
        x = np.concatenate((r,k),axis = 0)
        res = minimize(obj_scipy_inspect, x, method='SLSQP', jac=obj_grad_scipy_inspect, #Nelder-Mead #SLSQP
                   options={'ftol': 1e-9, 'maxiter':1000,'disp': True},bounds=mbounds) #constraints=ineq_cons, #,callback=output
    elif scenario == NOSCRAP:
        #Define Upper and Lower boundaries
        #The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])           
        mbounds = Bounds([smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue])          
        U_init = hp.U_noscrap(cost,USY,miuY,sigmaY_Taylor)
        x = np.copy(r)
        res = minimize(obj_scipy_noinspect, x, method='SLSQP', jac=obj_grad_scipy_noinspect, #Nelder-Mead #SLSQP
               options={'ftol': 1e-9, 'maxiter':1000,'disp': True},bounds=mbounds) #constraints=ineq_cons, #,callback=output
elif opt_lib == NLOPT:
    if scenario == SCRAP: #Scrap
        opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue])
        opt.set_min_objective(obj_nlopt_inspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.concatenate((r,k),axis = 0)
        x = opt.optimize(x0)
        U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1],x[2],x[3],x[4],x[5])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())        
    elif scenario == NOSCRAP:
        opt = nlopt.opt(nlopt.LD_MMA, m)
        opt.set_lower_bounds([smallvalue,smallvalue,smallvalue])
        opt.set_min_objective(obj_nlopt_noinspect)
        opt.set_xtol_rel(1e-4)
        x0 = np.copy(r)
        x = opt.optimize(x0)
        U_init = hp.U_noscrap(cost,USY,miuY,sigmaY_Taylor)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1],x[2])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())         



#Compare this method and CIRP method. 
        
if scenario == SCRAP: #Scrap
    ropt = x[0:m]
    kopt = x[m:]
    sigmaopt = hp.sigma(E,F,ropt)
    sigmacompare = np.array([0.0533, 0.09, 0.11])
    sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
    rcompare = hp.sigmator(sigmacompare,E,F)
    costcompare = hp.C(A,B,rcompare)
    kcompare = np.array([4.049, 1.668889, 4.6336])
    U_compare = hp.U_scrap(costcompare,USY,miuY,sigmaY_Taylorcompare,kcompare)   
    print('Old Method minimum value = ', U_compare )
elif scenario == NOSCRAP:
    ropt = x
    sigmaopt = hp.sigma(E,F,ropt)
    sigmacompare = np.array([0.0533, 0.09, 0.11])
    sigmaY_Taylorcompare = hp.sigmaY(sigmacompare,D)
    rcompare = hp.sigmator(sigmacompare,E,F)
    costcompare = hp.C(A,B,rcompare)
    U_compare = hp.U_noscrap(costcompare,USY,miuY,sigmaY_Taylorcompare)
    print('Old Method minimum value = ', U_compare )


from scipy.spatial import distance

def gradientcheck(x,scenario):
    #If Scipy is used
    if opt_lib == SCIPY:  
        if scenario == SCRAP:
            grad_equation = obj_grad_scipy_inspect(x) 
        
            #retrieve grad of r and k
            grad_equation_r = grad_equation[0:m]
            grad_equation_k = grad_equation[m:]  
            
            grad_numerical_k = np.zeros(m)  
            grad_numerical_r = np.zeros(m)
        elif scenario == NOSCRAP:
            grad_equation_r = obj_grad_scipy_noinspect(x) 
            grad_numerical_r = np.zeros(m)
            
        C = hp.C(A,B,r)
         
        for i in range(0,m):  
            ri_add_epsilon = np.copy(r)
            ri_minus_epsilon = np.copy(r)
            ri_add_epsilon[i] +=  epsilon
            ri_minus_epsilon[i] -=  epsilon
            ki_add_epsilon = np.copy(k)
            ki_minus_epsilon = np.copy(k)
            ki_add_epsilon[i] += epsilon
            ki_minus_epsilon[i] -= epsilon        
            sigmaX_plus = hp.sigma(E,F,ri_add_epsilon)  
            sigmaX_minus = hp.sigma(E,F,ri_minus_epsilon)  
            C_plus = hp.C(A,B,ri_add_epsilon)
            C_minus = hp.C(A,B,ri_minus_epsilon)
            sigmaY_Taylor_plus = hp.sigmaY(sigmaX_plus,D)
            sigmaY_Taylor_minus = hp.sigmaY(sigmaX_minus,D)
            
            if scenario == SCRAP:
                sigmaY_Taylor_p = lamada*sigmaY_Taylor            
                #Varify dr
                sigmaY_Taylor_plus *= lamada
                sigmaY_Taylor_minus *= lamada
                #gradient computed by numerical estimation
                grad_numerical_r[i] = (hp.U_scrap(C_plus,USY,miuY,sigmaY_Taylor_plus,k)
                - hp.U_scrap(C_minus,USY,miuY,sigmaY_Taylor_minus,k))/(2*epsilon)
                #varify dk
                grad_numerical_k[i] = (hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_add_epsilon)
                - hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_minus_epsilon))/(2*epsilon)        
                print('Numerical_scrap_'+'dr'+str(i),'=',grad_numerical_r[i])     
                print('Equation_scrap_'+'dr'+str(i),'=',grad_equation_r[i])
                print('Numerical_scrap_'+'dk'+str(i),'=',grad_numerical_k[i])          
                print('Equation_scrap_'+'dk'+str(i),'=',grad_equation_k[i]) 
                
            elif scenario == NOSCRAP:
                #gradient computed by numerical estimation
                grad_numerical_r[i] = (hp.U_noscrap(C_plus,USY,miuY,sigmaY_Taylor_plus) -
                              hp.U_noscrap(C_minus,USY,miuY,sigmaY_Taylor_minus))/(2*epsilon)
                print('Numerical_No scrap_'+'dr'+str(i),'=',grad_numerical_r[i])
                print('Equation_No scrap_'+'dr'+str(i),'=',grad_equation_r[i])            
         
        distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
        length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
        length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
        graderror_r = distance12_r/(length1_r + length2_r)
        print('error of dr=',graderror_r)
        
        if scenario == SCRAP:
            distance12_k =  distance.euclidean(grad_equation_k,grad_numerical_k)
            length1_k = distance.euclidean(grad_equation_k,np.zeros_like(grad_equation_k))
            length2_k = distance.euclidean(grad_numerical_k,np.zeros_like(grad_numerical_k))
            graderror_k = distance12_k/(length1_k + length2_k)
            print('error of dk=',graderror_k)  
            
    #if nlopt is used        
    elif opt_lib == NLOPT:  
        if scenario == SCRAP:
            grad_equation = np.zeros_like(x)
            obj_nlopt_inspect(x, grad_equation)
        
            #retrieve grad of r and k
            grad_equation_r = grad_equation[0:m]
            grad_equation_k = grad_equation[m:]  
            
            grad_numerical_k = np.zeros(m)  
            grad_numerical_r = np.zeros(m)
        elif scenario == NOSCRAP:
            grad_equation = np.zeros(m)
            obj_nlopt_noinspect(x,grad_equation)
            grad_equation_r = grad_equation[0:m]
            grad_numerical_r = np.zeros(m)
            
        C = hp.C(A,B,r)
         
        for i in range(0,m):  
            ri_add_epsilon = np.copy(r)
            ri_minus_epsilon = np.copy(r)
            ri_add_epsilon[i] +=  epsilon
            ri_minus_epsilon[i] -=  epsilon
            ki_add_epsilon = np.copy(k)
            ki_minus_epsilon = np.copy(k)
            ki_add_epsilon[i] += epsilon
            ki_minus_epsilon[i] -= epsilon        
            sigmaX_plus = hp.sigma(E,F,ri_add_epsilon)  
            sigmaX_minus = hp.sigma(E,F,ri_minus_epsilon)  
            C_plus = hp.C(A,B,ri_add_epsilon)
            C_minus = hp.C(A,B,ri_minus_epsilon)
            sigmaY_Taylor_plus = hp.sigmaY(sigmaX_plus,D)
            sigmaY_Taylor_minus = hp.sigmaY(sigmaX_minus,D)
            
            if scenario == SCRAP:
                sigmaY_Taylor_p = lamada*sigmaY_Taylor            
                #Varify dr
                sigmaY_Taylor_plus *= lamada
                sigmaY_Taylor_minus *= lamada
                #gradient computed by numerical estimation
                grad_numerical_r[i] = (hp.U_scrap(C_plus,USY,miuY,sigmaY_Taylor_plus,k)
                - hp.U_scrap(C_minus,USY,miuY,sigmaY_Taylor_minus,k))/(2*epsilon)
                #varify dk
                grad_numerical_k[i] = (hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_add_epsilon)
                - hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_minus_epsilon))/(2*epsilon)        
                print('Numerical_scrap_'+'dr'+str(i),'=',grad_numerical_r[i])     
                print('Equation_scrap_'+'dr'+str(i),'=',grad_equation_r[i])
                print('Numerical_scrap_'+'dk'+str(i),'=',grad_numerical_k[i])          
                print('Equation_scrap_'+'dk'+str(i),'=',grad_equation_k[i]) 
                
            elif scenario == NOSCRAP:
                #gradient computed by numerical estimation
                grad_numerical_r[i] = (hp.U_noscrap(C_plus,USY,miuY,sigmaY_Taylor_plus) -
                              hp.U_noscrap(C_minus,USY,miuY,sigmaY_Taylor_minus))/(2*epsilon)
                print('Numerical_No scrap_'+'dr'+str(i),'=',grad_numerical_r[i])
                print('Equation_No scrap_'+'dr'+str(i),'=',grad_equation_r[i])            
         
        distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
        length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
        length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
        graderror_r = distance12_r/(length1_r + length2_r)
        print('error of dr=',graderror_r)
        
        if scenario == SCRAP:
            distance12_k =  distance.euclidean(grad_equation_k,grad_numerical_k)
            length1_k = distance.euclidean(grad_equation_k,np.zeros_like(grad_equation_k))
            length2_k = distance.euclidean(grad_numerical_k,np.zeros_like(grad_numerical_k))
            graderror_k = distance12_k/(length1_k + length2_k)
            print('error of dk=',graderror_k)         

#Sometimes the gradients computed by equation may explode...
#gradientcheck(x,scenario)



