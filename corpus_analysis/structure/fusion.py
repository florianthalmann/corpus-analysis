import numpy as np
from ..stats.util import entropy

def entropy_fusion(matrices, size=10, resolution=10):
    return entropy_filter(matrices, size, resolution)

def entropy_filter(a, size, resolution):
    quant = np.around(a/np.max(a)*resolution).astype(int)
    strides = strides2D(quant, size)
    entropies = np.apply_along_axis(entropy, 2, strides)
    padded = np.pad(entropies, ((size-1,size-1),(size-1,size-1)), 'edge')
    meanents = np.mean(strides2D(padded, size), axis=2)
    return a*meanents

def strides2D(a, size):
    s,t = a.strides
    m,n = a.shape
    windows = np.lib.stride_tricks.as_strided(a,
        shape=(m-size+1, n-size+1, size, size), strides=(s,t,s,t))
    return np.reshape(windows, (m-size+1, n-size+1, -1))

#print(entropy_filter(np.array([[1,3,1,4],[2,3,4,1],[3,4,5,2]]), size=2))