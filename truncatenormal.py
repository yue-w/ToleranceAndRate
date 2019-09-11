# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:53:59 2019

@author: wyue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


N = 10000

krange = np.arange(0.01,4,0.01)

sample = np.random.normal(0,1,N)


sigma = np.std(sample)
print(sigma)



def sigmoid(x):
    return np.divide(np.ones_like(x),(1+np.exp(-1*x)))

sigmoidv = sigmoid(krange)
sigmoidvshift = 2*sigmoidv - 1



sigmapvec = np.zeros_like(krange)
for index, k in enumerate(krange):
    sigmap = 0
    for i in np.arange(1,4):
        sampletruncate = sample[np.logical_and(sample>(-k),sample<k)]
        sigmap += np.std(sampletruncate)
    sigmapvec[index] = sigmap/3

fig, ax1 = plt.subplots()
ax1.scatter(krange, sigmapvec, s=0.1, label="k curve", color='r' )
ax1.scatter(krange, sigmoidvshift, s=0.1, label="sigmoid", color='g' )
ax1.scatter(krange, erf(krange/np.sqrt(2)), s=0.1, label="error function", color='b' )
ax1.set_ylabel('sigmap') 
ax1.set_xlabel('k')
ax1.legend()
ax1.grid(True)
plt.show()

def plothist(sample):
    
    n, bins, patches = plt.hist(sample, 50, density=True, facecolor='g', alpha=0.75)
    
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram')
    plt.text(60, .025, r'$\mu=0,\ \sigma=1$')
    
    plt.grid(True)
    plt.show()


#plothist(sample)