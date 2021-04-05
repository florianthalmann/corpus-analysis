import numpy as np
from itertools import product
from ..util import plot_sequences, plot_matrix
from .histograms import freq_trans_hists
from .util import chiSquared

#removes all sequences that are more than the given stds away from the mean
def remove_outliers(sequences, stds=5):
    #plot_sequences(sequences, 'results5/-5.png')
    hists = freq_trans_hists(sequences, True, False)
    l = len(sequences)
    dists = np.array([chiSquared(hists[i], hists[j])
        for i,j in product(range(l), range(l))]).reshape((l,l))
    #plot_matrix(dists, 'results5/-7.png')
    means = np.mean(dists, axis=1)
    mean = np.mean(means)
    std = np.std(means)
    sequences = [s for i,s in enumerate(sequences) if (means[i]-mean) < stds*std]
    #plot_sequences(sequences, 'results5/-6.png')
    return sequences

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
