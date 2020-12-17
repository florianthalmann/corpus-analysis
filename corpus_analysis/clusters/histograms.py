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
    pairs = [np.vstack([s[:-1], s[1:]]).T for s in sequences]
    if ignore_same:
        pairs = [np.array([p for p in ps if p[0] != p[1]]) for ps in pairs]
    return frequency_histograms(pairs, relative)

#element frequency and non-equal transition frequency histogram
def freq_trans_hists(sequences, relative):
    return np.append(frequency_histograms(sequences, relative),
        transition_histograms(sequences, relative, True), axis=1)

#returns indices of clustered and unclustered hists
def clusters(hists):
    cluster_labels = OPTICS().fit(hists).labels_
    clustered = [i for i in range(len(hists)) if cluster_labels[i] != -1]
    unclustered = [i for i in range(len(hists)) if cluster_labels[i] == -1]
    return group_by(clustered, lambda i: cluster_labels[i]), unclustered

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