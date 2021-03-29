import numpy as np

def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))