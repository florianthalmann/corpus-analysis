import os, json, cProfile, math, tqdm
from functools import reduce
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from graph_tool.all import graph_draw

def multiprocess(title, func, data):
    print(title)
    with Pool(processes=cpu_count()-1) as pool:
        return list(tqdm.tqdm(pool.imap(func, data), total=len(data)))

def flatten(array, iterations=math.inf):#iterations -1 is deep flatten
    if iterations >= 0 and isinstance(array, list):
        return [b for a in array for b in flatten(a, iterations-1)]
    return [array]

def group_by(array, labels=None):
    if labels is None: labels = array
    uniq = np.unique(labels)
    return [array[labels == u] for u in uniq]

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

def plot_graph(graph, path):
    graph_draw(graph, output_size=(1000, 1000), output=path, bg_color=[1,1,1,1])

def plot_sequences(sequences, path=None):
    maxlen = max([len(s) for s in sequences])
    minval = np.min(np.hstack(sequences))
    #offset to 1 and pad with 0s
    matrix = np.vstack([np.pad(s-minval+1, (0, maxlen-len(s))) for s in sequences])
    hls = sns.color_palette("hls", np.max(matrix)+1)
    if np.min(matrix) == 0: hls[0] = (1,1,1)
    plt.figure(figsize=(6.4,4.8))#somehow heatmap gets squished otherwise
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap=hls)
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def plot(data, path=None):
    plt.plot(data)
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def boxplot(data, path=None):
    pd.DataFrame(data).boxplot()
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.clf()

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def buffered_run(name, func, params=[]):
    path = name+'_'.join(str(p) for p in params)+'.np'
    if os.path.isfile(path+'y'):
        return np.load(path+'y', allow_pickle=True)
    elif os.path.isfile(path+'z'):
        loaded = np.load(path+'z', allow_pickle=True)
        return [loaded[f] for f in loaded.files]
    data = func()
    try:
        np.save(path+'y', data)
    except ValueError:
        os.remove(path+'y')#empty file saved despite error
        np.savez_compressed(path+'z', *data)
    return data

def profile(func):
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    pr.print_stats()