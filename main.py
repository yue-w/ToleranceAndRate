# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:48:35 2019

@author: wyue
Optimizing production rate. Clutch Case Study
"""

#import packages
import numpy as np
import helpers as hp
import commonfunc as cf
from scipy.optimize import minimize
from scipy.optimize import Bounds

#number of components
m = 3

epsilon = 1e-7

#Nominal value of Y
miuY = np.radians(7.0124)
miu= np.array([55.291, 22.86, 101.6])

r = np.array([1.0, 1.0, 1.0])

CASE = 2
p = 0.95
(sigmaX_init1,sigmaX_init2, sigmaX_init3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)

sigmaX_init = np.array([sigmaX_init1, sigmaX_init2, sigmaX_init3])
TX = np.array([TX1,TX2,TX3])
k = np.divide(TX,sigmaX_init)

A = np.array([1.0, 2.0, 3.0])
B = np.array([1.0, 2.0, 3.0])
E = np.array([sigmaX_init1-1, sigmaX_init2-1, sigmaX_init3-1])
F = np.array([0.1, 1.0, 0.5])

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])


#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + np.radians(1.0)


lamada = 0.87794
    
#Concatenate r and k into a numpy array
x = np.concatenate((r,k),axis=0)

#Objective function
def objective(x):
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
def obj_grad(x):
    #retrieve r and k
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
    grad = np.concatenate((grad_r,grad_k),axis=0)
    return grad

    
#Define Upper and Lower boundaries
#The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])
smallvalue = 1e-2 # Lover bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 15
#mbounds = Bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue,largevalue,largevalue,largevalue])
mbounds = Bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue],[largevalue,largevalue,largevalue,largevalue,largevalue,largevalue])
#optimization vairable [r,k] and their initial values

#Define inequal constraint
ineq_cons = {'type': 'ineq',
              'fun' : lambda x: np.array([x[0], x[1]]),
              'jac' : lambda x: np.array([[1.0,0,0,0,0,0],[0,1.0,0,0,0,0]])} 

x = np.concatenate((r,k),axis = 0)

#Iteration count
ite = 1

def output(x):
    #Retrieve r and k
    global ite
    r = x[0:m]
    k = x[m:] 
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
U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k)
    
res = minimize(objective, x, method='SLSQP', jac=obj_grad, #Nelder-Mead #SLSQP
               options={'ftol': 1e-9, 'maxiter':1000,'disp': True},bounds=mbounds) #constraints=ineq_cons, #,callback=output


from scipy.spatial import distance

def gradientcheck_scrap(x):
    grad_equation = obj_grad(x) 

    #retrieve grad of r and k
    grad_equation_r = grad_equation[0:m]
    grad_equation_k = grad_equation[m:]  
    
    grad_numerical_k = np.zeros(m)  
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
     
    distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
    length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
    length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
    graderror_r = distance12_r/(length1_r + length2_r)
    print('error of dr=',graderror_r)
    
    distance12_k =  distance.euclidean(grad_equation_k,grad_numerical_k)
    length1_k = distance.euclidean(grad_equation_k,np.zeros_like(grad_equation_k))
    length2_k = distance.euclidean(grad_numerical_k,np.zeros_like(grad_numerical_k))
    graderror_k = distance12_k/(length1_k + length2_k)
    print('error of dk=',graderror_k)   

#gradientcheck_scrap(x)
