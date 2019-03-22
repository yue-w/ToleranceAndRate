# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:55:53 2019

@author: wyue
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import norm 
import helpers as hp
import commonfunc as cf

def plot(Y,Y_new,words,p):    
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
        
nsample = 1000
p = 0.995
miuX1 = 55.291
miuX2 = 22.86
miuX3 = 101.6
##Tolerance from Choi
#CASE = 1
##Tolerance from Zahara
CASE = 2
(sigmax1,sigmax2, sigmax3,TX1,TX2,TX3) = cf.init_sigmas(CASE,p)
    
dim_X1 = np.random.normal(loc=miuX1,scale=sigmax1,size=nsample)
dim_X2 = np.random.normal(loc=miuX2,scale=sigmax2,size=nsample)
dim_X3 = np.random.normal(loc=miuX3,scale=sigmax3,size=nsample)
dim_X = np.array([dim_X1,dim_X2,dim_X3])
Y = hp.assembly(dim_X)
Y2 = np.arccos((dim_X1+dim_X2)/(dim_X3-dim_X2))
Y_removeNAN = Y[np.logical_not(np.isnan(Y))]
plot(Y,Y_removeNAN, 'no scrap',p)
#save file to csv
hp.save_data_csv('Y.csv',Y)
hp.save_data_csv('Y_removeNAN.csv',Y_removeNAN)

(dim_X1_satis,N1) = hp.produce_satisfactory_output(miuX1, sigmax1, nsample, TX1)
(dim_X2_satis,N2) = hp.produce_satisfactory_output(miuX2, sigmax2, nsample, TX2)
(dim_X3_satis,N3) = hp.produce_satisfactory_output(miuX3, sigmax3, nsample, TX3)
dim_X_satis = np.array([dim_X1_satis,dim_X2_satis,dim_X3_satis])
Y_trumcate = hp.assembly(dim_X_satis)
Y_truncate_removeNAN = Y_trumcate[np.logical_not(np.isnan(Y_trumcate))]
#save file to csv
hp.save_data_csv('Y_trumcate.csv',Y_trumcate)
hp.save_data_csv('Y_truncate_removeNAN.csv',Y_truncate_removeNAN)
plot(Y_trumcate,Y_truncate_removeNAN, 'with scrap',p)
