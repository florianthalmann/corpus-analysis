import numpy as np
from sklearn.cluster import OPTICS
from ..util import group_by

def frequency_histogram(sequence, num_bins, relative):
    groups = group_by(sequence)
    hist = np.zeros(num_bins)
    for g in groups:
        hist[g[0]] = len(list(g))
    if relative:
        sum = np.sum(hist)
        return hist/sum if sum > 0 else hist
    return hist

def to_uniq_ids(sequences):
    sequences = [s[:,None] if s.ndim == 1 else s for s in sequences]
    if any(len(s) > 0 for s in sequences):
        uniq = np.unique([e for s in sequences for e in s], axis=0)
        to_index = {tuple(u):i for i,u in enumerate(uniq)}
        index_seqs = [np.array([to_index[tuple(p)] for p in ps]) for ps in sequences]
        return index_seqs, len(uniq)
    return sequences, 0

def frequency_histograms(sequences, relative):
    idseqs, num_ids = to_uniq_ids(sequences)
    return np.array([frequency_histogram(s, num_ids, relative) for s in idseqs])

def transition_histograms(sequences, relative, ignore_same=False):
    return tuple_histograms(sequences, relative, 2, ignore_same)

def tuple_histograms(sequences, relative, size=2, ignore_uniform=False):
    tuples = [np.vstack([s[i:-size+i+1] if i < size-1 else s[i:]
        for i in range(size)]).T for s in sequences]
    if ignore_uniform and size > 1:
        tuples = [np.array([t for t in ts if np.all(t == t[0])]) for ts in tuples]
    return frequency_histograms(tuples, relative)

#element frequency and non-equal transition frequency histogram
def freq_trans_hists(sequences, relative, ignore_same=False):
    hists = np.append(frequency_histograms(sequences, relative),
        transition_histograms(sequences, relative, ignore_same), axis=1)
    if relative:
        hists = np.array([h/np.sum(h) for h in hists])
    return hists

#returns indices of clustered and unclustered hists
def clusters(hists):
    if (len(hists) > 1):
        cluster_labels = OPTICS(min_samples=min(len(hists),5)).fit(hists).labels_
        clustered = [i for i in range(len(hists)) if cluster_labels[i] != -1]
        unclustered = [i for i in range(len(hists)) if cluster_labels[i] == -1]
        return group_by(clustered, lambda i: cluster_labels[i]), unclustered
    return [[0]], []

#classification of sequences of different lengths using frequency histograms
def freq_hist_clusters(sequences, relative=True):
    return clusters(frequency_histograms(sequences, relative))

def trans_hist_clusters(sequences, relative=True):
    return clusters(transition_histograms(sequences, relative))

def freq_trans_hist_clusters(sequences, relative=True):
    return clusters(freq_trans_hists(sequences, relative))

# print(freq_trans_hists([np.array([2,2,2,2]),np.array([1,1,1,1]),
#      np.array([3,3])], True))
# freq_trans_hists([np.array([1,2,1,2,2,1,1,2,1]),np.array([1,2,3,1,2]),
#     np.array([2,2,2])], True)