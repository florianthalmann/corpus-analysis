from itertools import product
import numpy as np
from ..alignment.affinity import get_alignment_segments, get_longest_segment
from ..alignment.smith_waterman import smith_waterman
from ..util import plot_sequences, plot_matrix, multiprocess, odd
from .histograms import freq_trans_hists, frequency_histograms, tuple_histograms
from .util import chiSquared

def check_double_time2(sequences, beats, window=4):
    tempos = [60/np.mean(b[1:]-b[:-1]) for b in beats]
    doubles = np.array([odd(s) for s in sequences])
    halves = np.array([np.repeat(s, 2) for s in sequences], dtype='object')
    
    btempos = best_combo([[np.array([-1,0,1])[np.argmin(np.abs([u/2-t,u-t,2*u-t]))]
        for u in tempos] for t in tempos])
    
    stacked = np.hstack((sequences, doubles, halves))
    relative = False
    ignore_uniform = False
    hists = [best_hist_combo(tuple_histograms(stacked, relative, i, ignore_uniform))
        for i in range(1, 8, 2)]
    b = np.mean(np.vstack((btempos, *hists)), axis=0)
    plot_matrix(np.vstack((btempos, *hists)), 'results/*-.png')
    t = 0.8
    print('adjusted', [b for b in [(i,-1) if b[i] <= -t else (i,1) if b[i] >= t else None
        for i,s in enumerate(sequences)] if b is not None])
    sequences = [halves[i] if b[i] <= -t else doubles[i] if b[i] >= t else s
        for i,s in enumerate(sequences)]
    beats = [interpolate(s) if b[i] <= -t else odd(s) if b[i] >= t else s
        for i,s in enumerate(beats)]
    return sequences, beats

#returns a copy of a interpolated with means and with an added interval equal to the last
def interpolate(a):
    means = np.mean(np.vstack((a[:-1],a[1:])), axis=0)
    means = np.insert(means, len(means), a[-1]+(a[-1]-means[-1]))
    return np.vstack((a,means)).reshape((-1,), order='F')

def check_double_time(sequences, pattern_count=50):
    #plot_sequences(sequences, 'results5/-1.png')
    doubles = np.array([odd(s) for s in sequences])
    halves = np.array([np.repeat(s, 2) for s in sequences], dtype='object')
    #WHY TWO STEPS AT A TIME??
    b = best_with_feature_freq(sequences, doubles, halves)
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
    return best_hist_combo(hists)

COUNT = 50
def get_segments(sequence):
    if len(sequence) == 0: return []
    return get_alignment_segments(sequence, sequence, COUNT, 16, 1, 4, .2)

def best_with_feature_freq(sequences, doubles, halves):
    hists = freq_trans_hists(np.hstack((sequences, doubles, halves)), True, False)
    return best_hist_combo(hists)

def best_hist_combo(hists):
    l = int(len(hists)/3)
    dists = np.zeros((l, l))
    for i,j in product(range(l), range(l)):
        o = chiSquared(hists[i], hists[j])
        d = chiSquared(hists[i+l], hists[j])
        h = chiSquared(hists[i+(2*l)], hists[j])
        dists[i][j] = np.array([0,1,-1])[np.argmin([o,d,h])]
    #plot_matrix(dists, 'results5/-3.png')
    return best_combo(dists)

def best_combo(dist_matrix, threshold=0.33):
    means = (np.mean(dist_matrix, axis=1) - np.mean(dist_matrix, axis=0)) / 2
    return np.array([-1 if m <= -0.33 else 1 if m >= 0.33 else 0 for m in means])

#print(interpolate(np.array([1,3,6,7])))