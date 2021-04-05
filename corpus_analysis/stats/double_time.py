from itertools import product
import numpy as np
from ..alignment.affinity import get_alignment_segments, get_longest_segment
from ..alignment.smith_waterman import smith_waterman
from ..util import plot_sequences, plot_matrix, multiprocess
from .histograms import freq_trans_hists, frequency_histograms
from .util import chiSquared

def check_double_time(sequences, pattern_count=50):
    #plot_sequences(sequences, 'results5/-1.png')
    doubles = np.array([s[np.arange(round(len(s)/2))*2] for s in sequences])
    halves = np.array([np.repeat(s, 2) for s in sequences])
    b = best_with_feature_freq(sequences)
    b += best_with_patterns([halves[i] if b[i] == -1 else doubles[i] if b[i] == 1 else s
        for i,s in enumerate(sequences)], pattern_count)
    #b = best_with_patterns(sequences, pattern_count)
    sequences = [halves[i] if b[i] <= -1 else doubles[i] if b[i] >= 1 else s
        for i,s in enumerate(sequences)]
    #plot_sequences(sequences, 'results5/-2.png')
    return sequences

def best_with_patterns(sequences, pattern_count):
    COUNT = pattern_count
    sas = multiprocess('dt patterns', get_segments, sequences)
    intervals = np.array([np.array([s[0][1]-s[0][0] for s in a]) for a in sas])
    doubles = np.array([(i/2).astype(int) for i in intervals])
    halves = np.array([i*2 for i in intervals])
    hists = frequency_histograms(np.concatenate((intervals, doubles, halves)), True)
    return best_sequence_combo(sequences, hists)

COUNT = 50
def get_segments(sequence):
    return get_alignment_segments(sequence, sequence, COUNT, 16, 1, 4, .2)

def best_with_feature_freq(sequences):
    doubles = np.array([s[np.arange(round(len(s)/2))*2] for s in sequences])
    halves = np.array([np.repeat(s, 2) for s in sequences])
    hists = freq_trans_hists(np.hstack((sequences, doubles, halves)), True, False)
    return best_sequence_combo(sequences, hists)

def best_sequence_combo(sequences, hists):
    l = len(sequences)
    dists = np.zeros((l, l))
    for i,j in product(range(l), range(l)):
        o = chiSquared(hists[i], hists[j])
        d = chiSquared(hists[i+l], hists[j])
        h = chiSquared(hists[i+(2*l)], hists[j])
        dists[i][j] = np.array([0,1,-1])[np.argmin([o,d,h])]
    #plot_matrix(dists, 'results5/-3.png')
    means = (np.mean(dists, axis=1) - np.mean(dists, axis=0)) / 2
    return np.array([-1 if m <= -0.33 else 1 if m >= 0.33 else 0 for m in means])