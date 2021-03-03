import numpy as np
from .clusters.histograms import freq_trans_hists, frequency_histograms
from .util import plot_sequences
from matplotlib import pyplot as plt

def check_double_time(sequences):
    plot_sequences(sequences, 'results5/-1.png')
    doubles = np.array([s[np.arange(round(len(s)/2))*2] for s in sequences])
    halves = np.array([np.repeat(s, 2) for s in sequences])
    hists = freq_trans_hists(np.hstack((sequences, doubles, halves)), False)
    avg = np.mean(hists[:len(sequences)], axis=0)
    best = []
    for i in range(len(sequences)):
        ref = chiSquared(hists[i], avg)
        if chiSquared(hists[i+len(sequences)], avg) < ref:
            best.append(doubles[i])
        elif chiSquared(hists[i+(2*len(sequences))], avg) < ref:
            best.append(halves[i])
        else:
            best.append(sequences[i])
    plot_sequences(best, 'results5/-2.png')
    
    # plt.bar(np.arange(len(hists[0])), hists[0])
    # plt.savefig('results5/1.png', dpi=1000)
    # plt.close()
    # plt.bar(np.arange(len(hists[0])), hists[60])
    # plt.savefig('results5/2.png', dpi=1000)
    # plt.close()
    # plt.bar(np.arange(len(hists[0])), hists[-1])
    # plt.savefig('results5/3.png', dpi=1000)
    # plt.close()

def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))