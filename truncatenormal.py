# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:53:59 2019

@author: wyue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import norm 

N = 10000
averagetime = 30
originalsigma = 1

krange = np.arange(0.1,4,0.001)



def sigmoid(x):
    return np.divide(np.ones_like(x),(1+np.exp(-1*x)))

a = 2.26
c = 1.09
b = 1 + 1/np.exp(a*c)
d = -1/np.exp(a*c)

def sigmoidshift(x,a,b,c,d):
    #return np.divide(2*np.ones_like(x),1+np.exp(-1*a*x)) - 1
    return np.divide(b*np.ones_like(x),1+np.exp(-a*(x-c)))+d

#sigmoidv = sigmoid(krange)
sigmoidvshift = sigmoidshift(krange,a,b,c,d)

def sigmapsimulation(k):
    sigmap = 0
    for i in np.arange(1,averagetime+1):
        sample = np.random.normal(0,originalsigma,N)
        sampletruncate = sample[np.logical_and(sample>(-k),sample<k)]
        sigmap += np.std(sampletruncate)
    stdp =  sigmap/averagetime  
    return stdp

def W(k):
  #return np.divide(1,1-2*norm.cdf(-k))
  return np.divide(1,erf(k/np.sqrt(2)))

def phi(k):
  return np.divide(np.exp(-np.power(k,2)/2),np.sqrt(2*np.pi))

def spcbook(k):
  return np.sqrt(1+2*W(k)*(-k)*phi(-k))

def draw(krange):
    sigmapvec = np.zeros_like(krange)
    for index, k in enumerate(krange):
        sigmap = 0
        dev = k*originalsigma
        for i in np.arange(1,averagetime+1):
            sample = np.random.normal(0,originalsigma,N)
            sampletruncate = sample[np.logical_and(sample>(-dev),sample<dev)]
            sigmap += np.std(sampletruncate)/originalsigma
        sigmapvec[index] = sigmap/averagetime
    
    
    fig, ax1 = plt.subplots()
    ax1.scatter(krange, sigmapvec, s=5, label="k curve", color='r' )
    ax1.scatter(krange, sigmoidvshift, s=0.1, label="sigmoid", color='g' )
    #ax1.scatter(krange, erf(krange/np.sqrt(2)), s=0.1, label="error function", color='b' )
    #ax1.scatter(krange, np.tanh(krange), s=0.1, label="tanh", color='y' )
    spcbookks = spcbook(krange)
    ax1.scatter(krange, spcbookks, s=0.1, label="spcK", color='y' )
    ax1.set_ylabel('sigmap') 
    ax1.set_xlabel('k')
    ax1.legend()
    ax1.grid(True)
    plt.show()
    fig.savefig(fname='fittingk',dpi=300)

def plothist(sample):
    
    n, bins, patches = plt.hist(sample, 50, density=True, facecolor='g', alpha=0.75)
    
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram')
    plt.text(60, .025, r'$\mu=0,\ \sigma=1$')
    
    plt.grid(True)
    plt.show()


#plothist(sample)

draw(krange)


