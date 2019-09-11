# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

from functions import objective, obj_der

#import pdb



bounds = Bounds([-np.inf,0],[np.inf,np.inf]) #The order is ([lower bnd for x1, lower bnd for x2], [Higher bnd for x1, Higher bnd for x2])

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[1] - 8*(x[0]**3),
                                         x[1] -(1-x[0])**3]),
             'jac' : lambda x: np.array([[-24*x[0]**2, 1], #df1/dx0 , df1/dx1
                                         [3*((1-x[0])**2), 1]])}  #df2/dx0 , df2/dx1
    
x0 = np.array([1.234, 0.296296])
#x0 = np.array([0.333334, 0.3])
res = minimize(objective, x0, method='SLSQP', jac=obj_der,
               constraints=ineq_cons, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)


    