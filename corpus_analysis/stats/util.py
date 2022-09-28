import math
import numpy as np
from numpy.lib.stride_tricks import as_strided

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

def tempo(beats):
    return np.array([60/np.mean(b[1:]-b[:-1]) for b in beats])

def normalize(a):
    min, max = np.min(a), np.max(a)
    return (a-min)/(max-min)

def subsequences(arr, m):
    if arr.size <= m: return np.array([arr])
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s))
