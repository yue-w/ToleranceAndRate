

import numpy as np

# objective function 
def objective(x):
    #pdb.set_trace() #debug
    x2 = x[1]
    return np.sqrt(x2)

# Derivative of objection function
def obj_der(x):
    der = np.zeros_like(x)
    der[0] = 0.0;
    der[1] = 0.5 / np.sqrt(x[1]);

    return der

def testf():
    print('helper function')