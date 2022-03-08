import math
import numpy as np

def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))

def entropy(a):
    p = np.bincount(a)/len(a)
    #print(p)
    p = p[np.nonzero(p)]
    #print(p)
    return -np.sum(np.log2(p)*p)

def entropy2(a, base=None):
    value,counts = np.unique(a, return_counts=True)
    norm_counts = counts / counts.sum()
    base = math.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()