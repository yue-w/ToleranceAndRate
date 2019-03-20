# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:55:53 2019

@author: wyue
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import helpers as hp

def plot(Y,Y_new,words):    
    if Y_new.shape==Y.shape:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(Y,plot=plt)
        pstr = 'p='+str(p)
        title = 'Probplot, ' + words + ' (no error in assembly) ' + pstr
        ax.set_title(title)
    else:
        print('There are problems in computing arccos')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(Y_new,plot=plt)
        pstr = 'p='+str(p)
        title = 'Probplot, ' + words + '(err in assembly), ' + pstr 
        ax.set_title(title)   
        
nsample = 10000

X1 = 55.291
X2 = 22.86
X3 = 101.6


CASE = 1

if CASE == 1:
    # # # #Tolerance from Choi
    TX1 = 0.1798
    TX2 = 0.1653
    TX3 = 0.2  
elif CASE == 2:
    # #Tolerance from Zahara
    TX1 = 0.107696
    TX2 = 0.0127
    TX3 = 0.068072        

p = 0.995
p_alpha = 1-(1-p)/2
z = norm.ppf(p_alpha)


# =============================================================================
##Sigma from Choi
# sigmax1 = 0.059935 
# sigmax2 = 0.040044 
# sigmax3 = 0.066860
# =============================================================================
sigmax1 = TX1/z
sigmax2 = TX2/z
sigmax3 = TX3/z
dim_X1 = np.random.normal(loc=X1,scale=sigmax1,size=nsample)
dim_X2 = np.random.normal(loc=X2,scale=sigmax2,size=nsample)
dim_X3 = np.random.normal(loc=X3,scale=sigmax3,size=nsample)
dim_X = np.array([dim_X1,dim_X2,dim_X3])
Y = hp.assembly(dim_X)
Y2 = np.arccos((dim_X1+dim_X2)/(dim_X3-dim_X2))

Y_new = Y[np.logical_not(np.isnan(Y))]

plot(Y,Y_new, 'no scrap')

(dim_X1_satis,N1) = hp.produce_satisfactory_output(X1, sigmax1, nsample, TX1)
(dim_X2_satis,N2) = hp.produce_satisfactory_output(X2, sigmax2, nsample, TX2)
(dim_X3_satis,N3) = hp.produce_satisfactory_output(X3, sigmax3, nsample, TX3)
dim_X_satis = np.array([dim_X1_satis,dim_X2_satis,dim_X3_satis])
Y_trumcate = hp.assembly(dim_X_satis)

Y_new_truncate = Y_trumcate[np.logical_not(np.isnan(Y_trumcate))]

plot(Y_trumcate,Y_new_truncate, 'with scrap')
