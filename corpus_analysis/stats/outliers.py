import numpy as np
from itertools import product
from ..util import plot_sequences, plot_matrix
from .histograms import tuple_histograms, clusters, get_onset_hists
from .util import chiSquared, tempo

def split_into_songs(sequences):
    relative = tuple_histograms(sequences, True, 1)
    #absolute = tuple_histograms(sequences, False, 1)
    clustered, unclustered = clusters(relative, chiSquared)
    #print(clustered, unclustered)
    if len(unclustered) < len(sequences)/2:
        print('FOUND', len(clustered), 'CLUSTERS')
        return clustered, unclustered
    return [range(len(sequences))], []#one big cluster

#removes all sequences that are more than mindev stds away from the mean in at least numdevs features
def remove_outliers(sequences, beats, onsets, mindev=2.5, numdevs=2):
    #hists = [get_outlier_hists(tuple_histograms(sequences, True, 1))]
    stds = [get_outliers(tuple_histograms(sequences, True, 2)),
        get_outliers(tuple_histograms(sequences, True, 4)),
        get_outliers(get_onset_hists(beats, onsets, 64)),
        get_outliers(tempo(beats)),
        #get_outliers(get_onset_hists(beats, onsets, 10)),
        get_outliers([len(b) for b in beats])#duration outliers
        ]
    for s in stds:
        print(np.around(s, decimals=1))
    #b = np.min(np.vstack(hists), axis=0)
    #b = np.mean(np.vstack(hists), axis=0)
    #b = np.max(np.vstack(hists), axis=0)
    stds = np.vstack(stds).T
    #print(b)
    ngt2 = np.array([len(np.where(s > mindev)[0]) for s in stds])
    ngt4 = np.array([len(np.where(s > 4)[0]) for s in stds])
    print(ngt2)
    print(ngt4)
    
    plot_matrix(np.vstack(stds), 'results/*-o.png')
    removed = [i for i in range(len(sequences)) if ngt2[i] >= numdevs or ngt4[i] >= 1]#deviation more than x stds
    sequences = [s for i,s in enumerate(sequences) if i not in removed]
    return sequences, removed

def get_outliers(hists, std_threshold=3):
    l = len(hists)
    dists = np.array([chiSquared(hists[i], hists[j])
        for i,j in product(range(l), range(l))]).reshape((l,l))
    plot_matrix(dists, 'results/*-dists.png')
    means = np.mean(dists, axis=1)
    mean = np.mean(dists)
    std = np.std(dists)
    # return [1 if (means[i]-mean) >= std_threshold*std else 0
    #     for i in range(len(hists))]
    return [np.absolute((means[i]-mean)/std) for i in range(len(hists))]

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
