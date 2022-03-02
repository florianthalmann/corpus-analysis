import numpy as np
from itertools import product
from ..util import plot_sequences, plot_matrix
from .histograms import tuple_histograms
from .util import chiSquared

#removes all sequences that are more than the given stds away from the mean
def remove_outliers(sequences):
    hists = [get_outlier_hists(tuple_histograms(sequences, False, i, False))
        for i in range(1, 8, 2)]
    b = np.mean(np.vstack(hists), axis=0)
    plot_matrix(np.vstack(hists), 'results/*-o.png')
    removed = [i for i,s in enumerate(b) if s > 3] #> 0.5]
    sequences = [s for i,s in enumerate(sequences) if i not in removed]
    return sequences, removed

def get_outlier_hists(hists, std_threshold=3):
    l = len(hists)
    dists = np.array([chiSquared(hists[i], hists[j])
        for i,j in product(range(l), range(l))]).reshape((l,l))
    means = np.mean(dists, axis=1)
    mean = np.mean(means)
    std = np.std(means)
    # return [1 if (means[i]-mean) >= std_threshold*std else 0
    #     for i in range(len(hists))]
    return [(means[i]-mean)/std for i in range(len(hists))]

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
