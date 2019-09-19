8# -*- coding: utf-8 -*-
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

#r = np.array([1.0, 1.0, 1.0])

scenario = 3


#sigmaX = np.array([sigmax1, sigmax2, sigmax3])
#TX = np.array([TX1,TX2,TX3])

# =============================================================================
# A = np.array([0.87, 1.71, 3.54])#np.array([5.0, 3.0, 1.0])
# B = np.array([2.062, 1.276, 1.965]) #np.array([20.0, 36.7, 36.0])
# F = np.array([0.001798/3, 0.001653/3, 0.002/3])
# #E =  sigmaX_init - np.multiply(F,np.power(r,2)) #np.array([sigmaX_init1-1, sigmaX_init2-1, sigmaX_init3-1])
# E = np.array([0.083,0.096,0.129])/100
# =============================================================================


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

#r=hp.sigmator(sigmaX,E,F)
r = 10 * np.random.rand(3)
k = 5 * np.random.rand(3)#3 * np.ones_like(r) #

sigmaX = hp.sigma(E,F,r)
#Compute Unit Cost of initial value
C = hp.C(A,B,r)


sigmaY_Taylor = hp.sigmaY(sigmaX,D,scenario,k)

#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + np.radians(2.0)


#U = hp.U_scrap(C,USY,sigmaY,k)

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
    sigmaY_Taylor_plus = hp.sigmaY(sigmaX_plus,D,scenario,k)
    sigmaY_Taylor_minus = hp.sigmaY(sigmaX_minus,D,scenario,k)
    
   
    if scenario == 1: #NO INSPECT
        #gradient computed by numerical estimation
        grad_numerical_r[i] = (hp.U_noscrap(C_plus,USY,miuY,sigmaY_Taylor_plus,Sp) -
                      hp.U_noscrap(C_minus,USY,miuY,sigmaY_Taylor_minus,Sp))/(2*epsilon)
        print('Numerical_No scrap_'+'dr'+str(i),'=',grad_numerical_r[i])
        #gradient computed by equation
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)        
        grad_equation_r[i] = hp.dU_dri_noscrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp)
        print('Equation_No scrap_'+'dr'+str(i),'=',grad_equation_r[i])    
    
    elif scenario == 2: #Inspection FIX k             
        #Varify dr
        #gradient computed by numerical estimation
        grad_numerical_r[i] = (hp.U_scrap(C_plus,USY,miuY,sigmaY_Taylor_plus,k,Sp,Sc)
        - hp.U_scrap(C_minus,USY,miuY,sigmaY_Taylor_minus,k,Sp,Sc))/(2*epsilon)
        print('Numerical_scrap_'+'dr'+str(i),'=',grad_numerical_r[i])     
        #gradient computed by equation
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)
        grad_equation_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)
        print('Equation_scrap_'+'dr'+str(i),'=',grad_equation_r[i])
   
    elif scenario == 3: #Scrap and optimize k
        #Varify dr
        #gradient computed by numerical estimation
        grad_numerical_r[i] = (hp.U_scrap(C_plus,USY,miuY,sigmaY_Taylor_plus,k,Sp,Sc)
        - hp.U_scrap(C_minus,USY,miuY,sigmaY_Taylor_minus,k,Sp,Sc))/(2*epsilon)
        print('Numerical_scrap_'+'dr'+str(i),'=',grad_numerical_r[i])     
        #gradient computed by equation
        dCi_dri_v = hp.dCi_dri(B[i],r[i])
        dsigmai_dri_v = hp.dsigmai_dri(F[i],r[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX,r,i,dsigmai_dri_v,scenario,k)
        grad_equation_r[i] = hp.dU_dri_scrap(USY,miuY,sigmaY_Taylor,C,k,i,dsigmaY_dri_v,dCi_dri_v,Sp,Sc)        
        print('Equation_scrap_'+'dr'+str(i),'=',grad_equation_r[i])
        
        ##varify dk        
        ki_add_epsilon = np.copy(k)
        ki_minus_epsilon = np.copy(k)
        ki_add_epsilon[i] += epsilon
        ki_minus_epsilon[i] -= epsilon  
        sigmaY_Taylor_plus = hp.sigmaY(sigmaX,D,scenario,ki_add_epsilon)
        sigmaY_Taylor_minus = hp.sigmaY(sigmaX,D,scenario,ki_minus_epsilon)        
        grad_numerical_k[i] = (hp.U_scrap(C,USY,miuY,sigmaY_Taylor_plus,ki_add_epsilon,Sp,Sc)
        - hp.U_scrap(C,USY,miuY,sigmaY_Taylor_minus,ki_minus_epsilon,Sp,Sc))/(2*epsilon)
        print('Numerical_scrap_'+'dk'+str(i),'=',grad_numerical_k[i])     
        ##gradient computed by equation
        #grad_equation_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor_p,k[i],C[i],Sc[i])  
        dsigmaY_dki = hp.dsigmaY_dki(D,sigmaX,r,i,k)
        grad_equation_k[i] = hp.dU_dki_scrap(USY, miuY,sigmaY_Taylor,k,i,C,Sc,dsigmaY_dki,Sp)
        print('Equation_scrap_'+'dk'+str(i),'=',grad_equation_k[i])        
    
          

        
distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
graderror_r = distance12_r/(length1_r + length2_r)
print('error of dr=',graderror_r)

if scenario == 3: #INSPECT          
    distance12_k =  distance.euclidean(grad_equation_k,grad_numerical_k)
    length1_k = distance.euclidean(grad_equation_k,np.zeros_like(grad_equation_k))
    length2_k = distance.euclidean(grad_numerical_k,np.zeros_like(grad_numerical_k))
    graderror_k = distance12_k/(length1_k + length2_k)
    print('error of dk=',graderror_k)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    