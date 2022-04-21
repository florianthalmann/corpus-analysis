import numpy as np
from itertools import product
from ..util import plot_sequences, plot_matrix
from .histograms import tuple_histograms, clusters, get_onset_hists
from .util import chiSquared

def split_into_songs(sequences):
    relative = tuple_histograms(sequences, True, 1)
    #absolute = tuple_histograms(sequences, False, 1)
    clustered, unclustered = clusters(relative, chiSquared)
    #print(clustered, unclustered)
    if len(unclustered) < len(sequences)/2:
        print('FOUND', len(clustered), 'CLUSTERS')
        return clustered, unclustered
    return [range(len(sequences))], []#one big cluster

#removes all sequences that are more than the given stds away from the mean
def remove_outliers(sequences, beats, onsets):
    #hists = [get_outlier_hists(tuple_histograms(sequences, True, 1))]
    hists = [get_outliers(tuple_histograms(sequences, True, 2)),
        get_outliers(tuple_histograms(sequences, True, 4)),
        get_outliers(get_onset_hists(beats, onsets, 10)),
        get_outliers([b[-1]-b[0] for b in beats])]#duration outliers
    #b = np.min(np.vstack(hists), axis=0)
    b = np.mean(np.vstack(hists), axis=0)
    print(np.around(b, decimals=1))
    plot_matrix(np.vstack(hists), 'results/*-o.png')
    removed = [i for i,s in enumerate(b) if s > 2]#deviation more than x stds
    sequences = [s for i,s in enumerate(sequences) if i not in removed]
    return sequences, removed

def get_outliers(hists, std_threshold=3):
    l = len(hists)
    dists = np.array([chiSquared(hists[i], hists[j])
        for i,j in product(range(l), range(l))]).reshape((l,l))
    plot_matrix(dists, 'results/*-dists.png')
    means = np.mean(dists, axis=1)
    mean = np.mean(means)
    std = np.std(means)
    # return [1 if (means[i]-mean) >= std_threshold*std else 0
    #     for i in range(len(hists))]
    return [(means[i]-mean)/std for i in range(len(hists))]

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
