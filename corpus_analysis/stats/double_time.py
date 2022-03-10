from itertools import product
import numpy as np
from ..alignment.affinity import get_alignment_segments, get_longest_segment
from ..alignment.smith_waterman import smith_waterman
from ..util import plot_sequences, plot_matrix, multiprocess, odd
from .histograms import freq_trans_hists, frequency_histograms, tuple_histograms
from .util import chiSquared

def check_double_time2(sequences, beats, factors=None):
    factors = factors if factors is not None else [1 for s in sequences]
    sequences = [adjust_sequence(s, f) for s,f in zip(sequences, factors)]
    doubles = np.array([np.repeat(s, 2) for s in sequences], dtype='object')
    halves = np.array([odd(s) for s in sequences])
    
    #absolute histograms are able to indicate double time
    stacked = np.hstack((sequences, doubles, halves))
    hists = [best_hist_combo(tuple_histograms(stacked, False, i))
        for i in [4,8]] #4,8 best for #2
    
    plot_matrix(np.vstack(hists), 'results/*-.png')
    best = np.mean(np.vstack(hists), axis=0)
    
    #certify factors with tempo: beat detection should never be off by more than a factor 2!
    t = 1 #all parts have to agree
    max_dev = 2 #max deviation from mean tempo
    tempos = [60/np.mean(b[1:]-b[:-1]) for b in beats]
    meantempo = np.mean([f*t for f,t in zip(factors, tempos)])#with current factors
    newfactors = [0.5 if b <= -t else 2 if b >= t else f
        for f,b in zip(factors, best)]
    # newfactors = [0.5*f if b <= -t else 2*f if b >= t else f
    #     for f,b in zip(factors, best)]
    newtempos = [f*t for f,t in zip(newfactors, tempos)]
    factors = [n if (1/max_dev)*meantempo <= t <= max_dev*meantempo else f
        for f,t,n in zip(factors, newtempos, newfactors)]
    
    # print('adjusted', [b for b in [(i,-1) if best[i] <= -t else (i,1) if best[i] >= t else None
    #     for i,s in enumerate(sequences)] if b is not None])
    sequences = [h if f < 1 else d if f > 1 else s
        for f,s,h,d in zip(factors, sequences, halves, doubles)]
    
    # beats = [interpolate(s) if b[i] <= -t else odd(s) if b[i] >= t else s
    #     for i,s in enumerate(beats)]
    return sequences, factors #don't reuse these sequences as quality may decline!

def adjust_sequence(sequence, factor):
    while factor > 1:
        sequence = np.repeat(sequence, 2)
        factor /= 2
    while factor < 1:
        sequence = odd(sequence)
        factor *= 2
    return sequence

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
    return np.array([-1 if m < -0.5 else 1 if m > 0.5 else 0 for m in means])

#print(interpolate(np.array([1,3,6,7])))