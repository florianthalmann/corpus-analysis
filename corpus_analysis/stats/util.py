import numpy as np

def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))

def entropy(a):
    p = np.bincount(a)/len(a)
    p = p[np.nonzero(p)]
    return -np.sum(np.log2(p)*p)