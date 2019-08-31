import functions as f

import numpy as np

import nlopt

a = 4

def testF():
    print(a)
    
def testF2():
    global a 
    a = 2
    testF()

testF2()    