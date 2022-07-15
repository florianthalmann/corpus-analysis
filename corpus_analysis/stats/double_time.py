import math
from itertools import product
import numpy as np
from ..alignment.affinity import get_alignment_segments, get_longest_segment
from ..alignment.smith_waterman import smith_waterman
from ..util import plot_sequences, plot_matrix, multiprocess, summarize1d, interpolate
from .histograms import freq_trans_hists, frequency_histograms,\
    tuple_histograms, get_onset_hists
from .util import chiSquared, tempo, normalize

def check_double_time2(sequences, beats, onsets, factors=None, tempo_range=[50, 90]):#[50, 170]
    MAX_TEMPO_DEV = 2 #max deviation from mean tempo
    if factors == None:#init factors in reasonable range
        tempos = tempo(beats)
        print([int(t) for t in tempos])
        factors = np.array([get_init_factor(t, tempo_range) for t in tempos])
        #factors = np.ones(len(sequences))
    currentseqs = np.array([adjust_sequence(s, f) for s,f in zip(sequences, factors)])
    doubles = np.array([np.repeat(s, 2) for s in currentseqs], dtype='object')
    halves = np.array([summarize1d(s, 2) for s in currentseqs])
    twothirds = np.array([summarize1d(s, 3) for s in doubles])
    #threehalves = np.array([summarize1d(np.repeat(s, 3), 2) for s in doubles])
    
    #absolute chord and onset histograms are able to indicate double time
    stacked = np.hstack((currentseqs, doubles, halves, twothirds))
    # hists = [best_hist_combo(tuple_histograms(stacked, False, i))
    #     for i in [4,8]] #4,8 best for #2
    # hists = [best_hist_combo(tuple_histograms(stacked, True, i))
    #     for i in [2,4]]#,4]] #4,8 best for #2
    #dists = [hist_dists(np.array_split(tuple_histograms(stacked, True, 2), 4))]
    dists = [hist_dists(np.array_split(tuple_histograms(stacked, True, 4), 4))]
    
    currentbeats = [adjust_beats(b, f) for b,f in zip(beats, factors)]
    beats2 = [interpolate(b) for b in currentbeats]
    beats12 = [b[::2] for b in currentbeats]
    beats23 = [b[::3] for b in beats2]
    
    dists += [hist_dists([get_onset_hists(onsets, bb, 64)
        for bb in [currentbeats, beats2, beats12, beats23]])]
    
    tempos = tempo(currentbeats)
    print([int(t) for t in tempos])
    t = np.reshape(tempos, (-1, 1))
    dists += [hist_dists([t, t*2, t/2, t*2/3])]
    
    dists = [normalize(d) for d in dists]
    
    for i in np.where((tempos > 100) | (tempos < 70))[0]:
        print(tempos[i], [np.round(h[i], decimals=2) for h in dists], np.mean([h[i] for h in dists], axis=0))
    
    
    #plot_matrix(np.vstack(hists), 'results/*-.png')
    #best = np.mean(np.vstack(hists), axis=0)
    
    #print(np.dstack(dists))
    
    #means = [np.mean(np.vstack([d[i] for d in dists]))]
    
    means = np.mean(np.dstack(dists), axis=2)
    #print(means)
    best = np.argmin(means, axis=1)
    print(best)
    
    #print(best)
    
    #certify factors with tempo: beat detection should never be off by more than a factor 2!
    #t = 0.35 #degree of agreement
    meantempo = np.mean(tempos)#with current factors
    print(meantempo)
    #change = np.array([0.5 if b <= -t else 2 if b >= t else 1 for b in best])
    change = np.array([np.array([1,2,0.5,2/3])[b] for b in best])
    newfactors, newtempos = factors*change, tempos*change
    # for i in np.where(best >= t)[0]:
    #     print(tempos[i], newtempos[i], factors[i], newfactors[i], [h[i] for h in hists])
    #print(newtempos)
    factors = [n if (1/MAX_TEMPO_DEV)*meantempo <= t <= MAX_TEMPO_DEV*meantempo
            and tempo_range[0] <= t <= tempo_range[1] else f
        for f,t,n in zip(factors, newtempos, newfactors)]
    #factors = newfactors
    # print('adjusted', [b for b in [(i,-1) if best[i] <= -t else (i,1) if best[i] >= t else None
    #     for i,s in enumerate(sequences)] if b is not None])
    sequences = [adjust_sequence(s, f) for s,f in zip(sequences, factors)]
    beats = [adjust_beats(b, f) for b,f in zip(beats, factors)]
    return factors, sequences, beats #don't reuse these sequences and beats as quality may decline!


def get_init_factor(tempo, tempo_range):
    factor = 1
    while tempo < tempo_range[0]:
        factor *= 2
        tempo *= 2
    while tempo > tempo_range[1]:
        factor /= 2
        tempo /= 2
    return factor

def adjust_sequence(sequence, factor):
    if math.isclose(factor, 2/3):
        sequence = summarize1d(np.repeat(sequence, 2), 3)
    while factor % 2 == 0:
        sequence = np.repeat(sequence, 2)
        factor /= 2
    while (1/factor) % 2 == 0:
        sequence = summarize1d(sequence, 2)
        factor *= 2
    while (1/factor) % 3 == 0:
        sequence = summarize1d(sequence, 3)
        factor *= 3
    return sequence

def adjust_beats(beats, factor):
    if math.isclose(factor, 2/3):
        beats = interpolate(beats)[::3]
    while factor % 2 == 0:
        beats = interpolate(beats)
        factor /= 2
    while (1/factor) % 2 == 0:
        beats = beats[::2]
        factor *= 2
    while (1/factor) % 3 == 0:
        beats = beats[::3]
        factor *= 3
    return beats

def check_double_time(sequences, pattern_count=50):
    #plot_sequences(sequences, 'results5/-1.png')
    doubles = np.array([summarize1d(s, 2) for s in sequences])
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

#returns the average distance of each option hist from all original hists
def hist_dists(hists):
    l = len(hists[0])
    dists = np.zeros((l, l, len(hists)))
    for i,j in product(range(l), range(l)):
        dists[i][j] = np.array([chiSquared(h[i], hists[0][j]) for h in hists])
    return np.mean(dists, axis=1)

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
#print(hist_dists(np.array([[[70],[154],[80]],[[140],[308],[160]],[[35],[77],[40]]])))