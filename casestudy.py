9# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:55:01 2019

@author: wyue
"""

from pandas import DataFrame
import numpy as np
from scipy.stats import norm 

E = np.array([0.036,0.0432,0.054]) * 1.0 
ratio = np.linspace(0.5,1.8,20)

tols = np.array([0.237, 0.171, 0.297])

#Satisfaction rate for all components of all scenarios
beta_all = np.zeros((len(ratio),len(tols))) 
for idx_rate, rate in enumerate(ratio):
    for idx_tol, tol in enumerate(tols):
        x = tol / (rate * E[idx_tol])
        px = norm.cdf(x)
        beta = 1-2*(1-px)
        beta_all[idx_rate,idx_tol] = beta


#print(beta_all) 
#Write to excel
df = DataFrame({'part1':beta_all[:,0],'part2':beta_all[:,1],'part3':beta_all[:,2]})
df.to_excel('beta.xlsx', sheet_name='sheet2', index=False)
        
        
        
        