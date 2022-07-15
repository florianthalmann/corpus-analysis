import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from corpus_analysis.util import flatten, plot_sequences, boxplot
from gd import get_versions_by_date, SONGS, get_beats,\
    get_chord_sequences
from main import get_preprocessed_seqs

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
    years_by_song = [[d.year for d in ds] for ds in dates]
    all_years = [d.year for d in flatten(dates)]
    all_years = np.arange(np.min(all_years), np.max(all_years)+1)
    fig, ax = plt.subplots(figsize=(7, 5.25))
    sum = np.zeros(len(all_years))
    for i,years in enumerate(years_by_song):
        ys, cs = np.unique(years, return_counts=True)
        counts = np.zeros(len(all_years), dtype=int)
        for y,c in zip(ys, cs):
            counts[np.where(all_years == y)] = c
        ax.bar(all_years, counts, label=SONGS[i], width=1, bottom=sum)
        sum += counts
    ax.legend(ncol=2)
    plt.xlabel('time')
    plt.ylabel('version count')
    plt.tight_layout()
    fig.savefig('results/gd1.pdf')
    #plt.show()

def plot_relative_date_histogram():
    dates = [get_versions_by_date(s)[1] for s in SONGS]
    years_by_song = [[d.year for d in ds] for ds in dates]
    all_years = [d.year for d in flatten(dates)]
    _, all_counts = np.unique(all_years, return_counts=True)
    all_years = np.arange(np.min(all_years), np.max(all_years)+1)
    fig, ax = plt.subplots(figsize=(7, 5.25))
    sum = np.zeros(len(all_years))
    for i,years in enumerate(years_by_song):
        ys, cs = np.unique(years, return_counts=True)
        counts = np.zeros(len(all_years), dtype=int)
        for y,c in zip(ys, cs):
            counts[np.where(all_years == y)] = c
        counts = counts / all_counts
        ax.bar(all_years, counts, label=SONGS[i], width=1, bottom=sum)
        sum += counts
    #ax.legend(ncol=2)
    plt.xlabel('time')
    plt.ylabel('proportion')
    plt.tight_layout()
    fig.savefig('results/gd1rel.pdf')
    #plt.show()

def save_current_pandas_plot(outfile):
    plt.tight_layout()
    plt.suptitle('')
    plt.savefig(outfile)
    plt.close()

def plot_msa_eval(path):
    data = pd.read_csv(path)
    #data['avgent'] = data['entropy'] / data['length']
    #data['norment'] = data['entropy'] * data['vertices'] / data['vertices'].max()
    fig, axes = plt.subplots(1,3, figsize=(7, 5.25))
    data.boxplot(by='method', column=['entropy','partition count','total points'],
        ax=axes, positions=[0,2,3,1])
    save_current_pandas_plot('gd3.pdf')

def plot_num_mutuals_eval(path):
    data = pd.read_csv(path)
    fig, axes = plt.subplots(1,3, figsize=(7, 5.25))
    data.boxplot(by='num_mutual', column=['entropy','partition count','total points'],
        ax=axes)
    #data.groupby(['num_mutual']).plot(x='num_mutual')
    save_current_pandas_plot('gd-mutuals.pdf')

def plot_features(raw=False):
    chord_func = get_chord_sequences if raw else get_preprocessed_seqs
    data = pd.DataFrame([], columns=['song','duration','tempo','chord count'])
    for s in tqdm.tqdm(SONGS):
        for b,c in zip(get_beats(s), chord_func(s)):
            data.loc[len(data)] = [s,
                b[-1], 60/np.mean(b[1:]-b[:-1]), len(np.unique(c))]
    data.boxplot(by='song', column=['duration'], rot=90)
    save_current_pandas_plot('gd4-.pdf')
    data.boxplot(by='song', column=['tempo'], rot=90)
    save_current_pandas_plot('gd7-.pdf')
    data.boxplot(by='song', column=['chord count'], rot=90)
    save_current_pandas_plot('gd6-.pdf')

#plot_msa_eval('eval.csv')
#plot_num_mutuals_eval('mutuals.csv')
#plot_features()
plot_date_histogram2()
plot_relative_date_histogram()