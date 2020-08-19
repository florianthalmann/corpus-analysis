import os, json, cProfile
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

def plot_matrix(matrix, path=None):
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap=sns.cm.rocket_r)
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def buffered_run(path, func):
    print(os.path.isfile(path))
    if os.path.isfile(path):
        return np.load(path, allow_pickle=True)
    data = func()
    np.save(path, data)
    return data

def profile(func):
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    pr.print_stats()