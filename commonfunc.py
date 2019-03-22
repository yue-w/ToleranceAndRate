# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:31:58 2019

@author: wyue
"""
from scipy.stats import norm 

def init_sigmas(CASE,p): 
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
    return (sigmax1,sigmax2, sigmax3,TX1,TX2,TX3)