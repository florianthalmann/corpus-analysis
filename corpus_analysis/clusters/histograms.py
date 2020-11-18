import numpy as np
from sklearn.cluster import OPTICS
from ..util import group_by

def frequency_histogram(sequence, max_value, relative):
    groups = group_by(sequence)
    hist = np.zeros(max_value+1)
    for g in groups:
        hist[g[0]] = len(list(g))
    return hist/np.sum(hist) if relative else hist

#classification of sequences of different lengths using frequency histograms
def freq_hist_clusters(sequences, relative=True):
    max = np.amax(np.hstack(sequences))
    hists = [frequency_histogram(s, max, relative) for s in sequences]
    clustering = OPTICS().fit(hists)
    return clustering.labels_