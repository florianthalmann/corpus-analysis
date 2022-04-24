from itertools import product
import numpy as np
from ..alignment.affinity import get_alignment_segments, get_longest_segment
from ..alignment.smith_waterman import smith_waterman
from ..util import plot_sequences, plot_matrix, multiprocess, odd, interpolate
from .histograms import freq_trans_hists, frequency_histograms,\
    tuple_histograms, get_onset_hists
from .util import chiSquared, tempo

def check_double_time2(sequences, beats, onsets, factors=None):
    MAX_TEMPO_DEV = 1.5 #max deviation from mean tempo
    TEMPO_RANGE = [60, 180] #reasonable tempo values in pop music
    if factors == None:#init factors in reasonable range
        tempos = tempo(beats)
        print([int(t) for t in tempos])
        print(len(beats), len(sequences))
        factors = np.array([2 if t < TEMPO_RANGE[0] else 0.5 if t > TEMPO_RANGE[1]
            else 1 for t in tempos])
    currentseqs = [adjust_sequence(s, f) for s,f in zip(sequences, factors)]
    doubles = np.array([np.repeat(s, 2) for s in currentseqs], dtype='object')
    halves = np.array([odd(s) for s in currentseqs])
    
    #absolute chord and onset histograms are able to indicate double time
    stacked = np.hstack((currentseqs, doubles, halves))
    # hists = [best_hist_combo(tuple_histograms(stacked, False, i))
    #     for i in [4,8]] #4,8 best for #2
    hists = [best_hist_combo(tuple_histograms(stacked, True, i))
        for i in [2,4]]#,4]] #4,8 best for #2
    
    
    currentbeats = [adjust_beats(b, f) for b,f in zip(beats, factors)]
    dbeats = [interpolate(b) for b in currentbeats]
    hbeats = [odd(b) for b in currentbeats]
    hists += [best_hist_combo(np.concatenate([get_onset_hists(onsets, bb, 64)
        for bb in [currentbeats, dbeats, hbeats]]))]
    
    tempos = tempo(currentbeats)
    print([int(t) for t in tempos])
    t = np.reshape(tempos, (-1, 1))
    hists += [best_hist_combo(np.concatenate([t, t*2, t/2]))]
    
    # for i in np.where((tempos > 50) | (tempos < 75))[0]:
    #     print(tempos[i], [np.round(h[i], decimals=2) for h in hists], np.mean([h[i] for h in hists]))
    
    
    plot_matrix(np.vstack(hists), 'results/*-.png')
    best = np.mean(np.vstack(hists), axis=0)
    #print(best)
    
    #certify factors with tempo: beat detection should never be off by more than a factor 2!
    t = 0.35 #degree of agreement
    meantempo = np.mean(tempos)#with current factors
    print(meantempo)
    change = np.array([0.5 if b <= -t else 2 if b >= t else 1 for b in best])
    newfactors, newtempos = factors*change, tempos*change
    # for i in np.where(best >= t)[0]:
    #     print(tempos[i], newtempos[i], factors[i], newfactors[i], [h[i] for h in hists])
    #print(newtempos)
    factors = [n if (1/MAX_TEMPO_DEV)*meantempo <= t <= MAX_TEMPO_DEV*meantempo
            and TEMPO_RANGE[0] <= t <= TEMPO_RANGE[1] else f
        for f,t,n in zip(factors, newtempos, newfactors)]
    #factors = newfactors
    # print('adjusted', [b for b in [(i,-1) if best[i] <= -t else (i,1) if best[i] >= t else None
    #     for i,s in enumerate(sequences)] if b is not None])
    sequences = [adjust_sequence(s, f) for s,f in zip(sequences, factors)]
    beats = [adjust_beats(b, f) for b,f in zip(beats, factors)]
    return factors, sequences, beats #don't reuse these sequences and beats as quality may decline!

def adjust_sequence(sequence, factor):
    while factor > 1:
        sequence = np.repeat(sequence, 2)
        factor /= 2
    while factor < 1:
        sequence = odd(sequence)
        factor *= 2
    return sequence

def adjust_beats(beats, factor):
    while factor > 1:
        beats = interpolate(beats)
        factor /= 2
    while factor < 1:
        beats = odd(beats)
        factor *= 2
    return beats

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
    return (np.mean(dists, axis=1) - np.mean(dists, axis=0)) / 2#best_combo(dists)

def best_combo(dist_matrix, threshold=0.33):
    means = (np.mean(dist_matrix, axis=1) - np.mean(dist_matrix, axis=0)) / 2
    return np.array([-1 if m < -threshold else 1 if m > threshold else 0 for m in means])

#print(interpolate(np.array([1,3,6,7])))