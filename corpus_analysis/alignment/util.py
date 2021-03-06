import numpy as np

def strided(a, L, S=1):  # Window len = L, Stride len/stepsize = S
    if len(a) <= L: return np.array([a])
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def median_filter(a, radius):
    windows = strided(a, (radius*2)+1)
    medians = np.median(windows, axis=1)
    filtered = a.copy()
    filtered[radius:-radius] = medians[:]
    return filtered

def symmetric(A):
    return np.all(np.abs(A-A.T) == 0) if A.shape[0] == A.shape[1] else False

# print(np.array([0,0,0,0,1,0,1,1,0,0,1,1,0,0,0]))
# print(median_filter(np.array([0,0,0,0,1,0,1,1,0,0,1,1,0,0,0]), 2))