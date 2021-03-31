import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from corpus_analysis.util import flatten, plot_sequences, boxplot
from gd import get_versions_by_date, get_essentias

def plot_msa(song, sequences, msa):
    outseqs = [np.repeat(-1, len(s)) for s in sequences]
    for i,a in enumerate(msa):
        for j,m in enumerate(a):
            if len(m) > 0:
                outseqs[i][j] = int(m[1:])
    plot_sequences(outseqs, song+'-msa.png')

def plot_date_histogram():
    dates = [get_versions_by_date(s)[1] for s in SONGS]
    all_years = [d.year for d in flatten(dates)]
    labels, counts = np.unique(all_years, return_counts=True)
    plt.bar(labels, counts, align='center', width=1)
    plt.show()

def plot_date_histogram2():
    dates = [get_versions_by_date(s)[1] for s in SONGS]
    all_years = [d.year for d in flatten(dates)]
    years = [[d.year for d in ds] for ds in dates]
    fig, ax = plt.subplots(figsize=(7, 5.25))
    labels = np.arange(np.min(all_years), np.max(all_years)+1)
    sum = np.zeros(len(labels))
    for i,y in enumerate(years):
        l, counts = np.unique(all_years, return_counts=True)
        counts = np.pad(counts, (l[0]-labels[0], labels[-1]-l[-1]))
        ax.bar(labels, counts, label=SONGS[i], width=1, bottom=sum)
        sum += counts
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig('gd1.pdf')
    #plt.show()

def plot_msa_eval(path):
    data = pd.read_csv(path)
    #data['avgent'] = data['entropy'] / data['length']
    #data['norment'] = data['entropy'] * data['vertices'] / data['vertices'].max()
    fig, axes = plt.subplots(1,3, figsize=(7, 5.25))
    data.boxplot(by='method', column=['entropy','partition count','total points'],
        ax=axes, positions=[0,2,3,1])
    plt.tight_layout()
    plt.suptitle('')
    #plt.show()
    plt.savefig('gd3.pdf')

def plot_evolution(song):
    import matplotlib.pyplot as plt
    v, d = get_versions_by_date(song)
    #b = [e['rhythm']['onset_rate'] for e in get_essentias(SONGS[SONG_INDEX])]
    b = [e['lowlevel']['dynamic_complexity'] for e in get_essentias(song)]
    #b = [e['metadata']['audio_properties']['length'] for e in get_essentias(SONGS[SONG_INDEX])]
    #b = [b/2 if b > 140 else b for b in b]
    print(len(b))
    plt.plot(d, b)
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    b = running_mean(b, 10)
    print(len(b))
    b = np.pad(b, (5,4), 'constant', constant_values=(0,0))
    import matplotlib.pyplot as plt
    plt.plot(d, b)
    plt.show()

plot_msa_eval('eval.csv')