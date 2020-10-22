import os, json, cProfile, math
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def flatten(array, iterations=math.inf):#iterations -1 is deep flatten
    if iterations >= 0 and isinstance(array, list):
        return [b for a in array for b in flatten(a, iterations-1)]
    return [array]

def group_adjacent(numbers, max_dist=1):#groups adjacent numbers if within max_dist
    return np.array(reduce(
        lambda s,t: s+[[t]] if (len(s) == 0 or t-s[-1][-1] > max_dist)
            else s[:-1]+[s[-1]+[t]], numbers, []))

def mode(a, axis=0):
    values, counts = np.unique(a, axis=axis, return_counts=True)
    return values[counts.argmax()]

def ordered_unique(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]

def argmax(a):
    return max(enumerate(a), key=lambda x: x[1])[0]

def plot_matrix(matrix, path=None):
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap=sns.cm.rocket_r)
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def plot_hist(data, path=None, binwidth=1):
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def plot(data, path=None):
    plt.plot(data)
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def buffered_run(path, func):
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