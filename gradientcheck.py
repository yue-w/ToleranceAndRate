# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:54:11 2019
Check the gradient of ri and ki bomputed by equation and by numerical method
@author: wyue
"""

#import packages
import numpy as np
#import helper functions
import helpers as hp
#from scipy.stats import norm 
import commonfunc as cf
from scipy.spatial import distance

#number of components
m = 3

epsilon = 1e-7

#Nominal value of Y
miuY = np.radians(7.0124)

miu= np.array([55.291, 22.86, 101.6])

r = np.array([1.0, 1.0, 1.0])

CASE = 2
p = 0.95
(sigmax1,sigmax2, sigmax3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)

sigmaX = np.array([sigmax1, sigmax2, sigmax3])
TX = np.array([TX1,TX2,TX3])
k = np.divide(TX,sigmaX)

A = np.array([1.0, 2.0, 3.0])
B = np.array([1.0, 2.0, 3.0])
E = np.array([sigmax1-1, sigmax2-1, sigmax3-1])
F = np.array([1.0, 1.0, 1.0])

D1 = hp.dy_dx1(miu[0],miu[1],miu[2])
D2 = hp.dy_dx2(miu[0],miu[1],miu[2])
D3 = hp.dy_dx3(miu[0],miu[1],miu[2])

D = np.array([D1,D2,D3])

#Compute Unit Cost of initial value
C = hp.C(A,B,r)


sigmaY_Taylor = hp.sigmaY(sigmaX,D)

#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + np.radians(1.0)


#U = hp.U_scrap(C,USY,sigmaY,k)

lamada = 0.87794
scrap = 1


grad_numerical_r = np.zeros(m)
grad_equation_r = np.zeros(m)
grad_numerical_k = np.zeros(m)
grad_equation_k = np.zeros(m)
for i in range(0,m):  
    ri_add_epsilon = np.copy(r)
    ri_minus_epsilon = np.copy(r)
    ri_add_epsilon[i] +=  epsilon
    ri_minus_epsilon[i] -=  epsilon
    sigmaX_plus = hp.sigma(E,F,ri_add_epsilon)  
    sigmaX_minus = hp.sigma(E,F,ri_minus_epsilon)  
    C_plus = hp.C(A,B,ri_add_epsilon)
    C_minus = hp.C(A,B,ri_minus_epsilon)
    sigmaY_Taylor_plus = hp.sigmaY(sigmaX_plus,D)
    sigmaY_Taylor_minus = hp.sigmaY(sigmaX_minus,D)
    
    ki_add_epsilon = np.copy(k)
    ki_minus_epsilon = np.copy(k)
    ki_add_epsilon[i] += epsilon
    ki_minus_epsilon[i] -= epsilon
   
    if scrap == 1: #Scrap unsatisfactory components      
        sigmaY_Taylor_p = lamada*sigmaY_Taylor
        
        #Varify dr
        sigmaY_Taylor_plus *= lamada
        sigmaY_Taylor_minus *= lamada
        #gradient computed by numerical estimation
        grad_numerical_r[i] = (hp.U_scrap(C_plus,USY,miuY,sigmaY_Taylor_plus,k)
        - hp.U_scrap(C_minus,USY,miuY,sigmaY_Taylor_minus,k))/(2*epsilon)
        print('Numerical_scrap_'+'dr'+str(i),'=',grad_numerical_r[i])     
        #gradient computed by equation
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)
        grad_equation_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor_p,C,k,i,lamada,dsigmaY_dri_v,dCi_dri_v)
        print('Equation_scrap_'+'dr'+str(i),'=',grad_equation_r[i])
        
        #varify dk
        grad_numerical_k[i] = (hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_add_epsilon)
        - hp.U_scrap(C,USY,miuY,sigmaY_Taylor_p,ki_minus_epsilon))/(2*epsilon)
        print('Numerical_scrap_'+'dk'+str(i),'=',grad_numerical_k[i])     
        #gradient computed by equation
        grad_equation_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor_p,k[i],C[i])
        print('Equation_scrap_'+'dk'+str(i),'=',grad_equation_k[i])        
        

    elif scrap == 0: #No scrap
        #gradient computed by numerical estimation
        grad_numerical_r[i] = (hp.U_noscrap(C_plus,USY,miuY,sigmaY_Taylor_plus) -
                      hp.U_noscrap(C_minus,USY,miuY,sigmaY_Taylor_minus))/(2*epsilon)
        print('Numerical_No scrap_'+'dr'+str(i),'=',grad_numerical_r[i])
        #gradient computed by equation
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v)        
        grad_equation_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v)
        print('Equation_No scrap_'+'dr'+str(i),'=',grad_equation_r[i])

        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    