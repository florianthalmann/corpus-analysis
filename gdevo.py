import tqdm, operator
from functools import reduce
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks
from skmisc.loess import loess
import statsmodels.api as sm
import gd, main
from corpus_analysis import util, features
from corpus_analysis.stats.util import entropy2, tempo
from corpus_analysis.stats.histograms import get_onsetpos, get_onset_hists

PATH='results/gdevo/'

def plot_all():
    # plot_time()
    #plot_patterns()
    #plot_dynamics()
    plot_pitch()
    #plot_spectral()
    #plot_absolute_relative_comparison()

def plot_time():
    songs = gd.SONGS[:]
    chords, beats, dates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    tempos = [tempo(b) for b in beats]
    durations = [[b[-1]-b[0] for b in bs] for bs in beats]
    beatcounts = [[len(b) for b in bs] for bs in beats]
    beatdurs = [[b[1:]-b[:-1] for b in bs] for bs in beats]
    beatvars = [[np.std(b)/np.mean(b) for b in bs] for bs in beatdurs]
    plot_lowesses(dates, [tempos, durations, beatvars],
        ['tempo', 'duration', 'beatvar'], PATH+'*overall.lowess2.time.png')

def plot_patterns():
    songs = gd.SONGS[:]
    print("getting preprocessed sequences")
    chords, beats, dates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    versions = [np.setdiff1d(np.arange(len(c)+len(r)), r)
        for c,r in zip(chords, removed)]
    print("getting chroma and mfcc")
    chroma = [gd.get_chroma_sequences(*z) for z in zip(songs, versions, beats)]
    mfcc = [gd.get_mfcc_sequences(*z) for z in zip(songs, versions, beats)]
    print("calculating patterns")
    chromapats = to_patterns(chroma)
    mfccpats = to_patterns(mfcc)
    chromacount = [[len(np.unique(c, axis=0)) for c in cs] for cs in chromapats]
    mfcccount = [[len(np.unique(m, axis=0)) for m in ms] for ms in mfccpats]
    chordcount = [[len(np.unique(c)) for c in cs] for cs in chords]
    chromavar = [[freq_variation(c) for c in cs] for cs in chromapats]
    mfccvar = [[freq_variation(m) for m in ms] for ms in mfccpats]
    chordvar = [[freq_variation(c) for c in cs] for cs in chords]
    # chordvars = [[freq_variation(c) for c in cs] for cs in chords]
    # chordents = [[entropy2(c) for c in cs] for cs in chords]
    # plot_lowesses(dates, [chromacount, mfcccount, chordcount],
    #     ['chroma', 'mfcc', 'chord'], PATH+'*overall.lowess.patterns.png')
    print("plotting")
    ax = plot_lowesses(dates, [chromavar, mfccvar, chordvar],
        ['chroma', 'mfcc', 'chord'])#, PATH+'*overall.lowess2.patternvars.png')
    ax.legend(loc='upper left')
    
    years = np.concatenate(dates).astype('datetime64[Y]')
    counts = np.unique(years.astype(int), axis=0, return_counts=True)[1]
    ax = ax.twinx()
    ax.plot(np.unique(years), lowess(np.unique(years), counts/np.max(counts)),
        label='versions', color='red')
    ax.legend()
    plot(PATH+'*overall.lowess2.patternvars2.png')

def plot_dynamics():
    dates, essentias = get_essentias()
    
    loudness = get_essentia(essentias, ['lowlevel','loudness_ebu128','short_term','median'])
    loudness = [db_to_amp(l) for l in loudness]
    lstd = get_essentia(essentias, ['lowlevel','loudness_ebu128','short_term','stdev'])
    lmean = get_essentia(essentias, ['lowlevel','loudness_ebu128','short_term','mean'])
    lvarcoeff = [db_to_amp(s)/db_to_amp(m) for s,m in zip(lstd, lmean)]
    dycomp = get_essentia(essentias, ['lowlevel','dynamic_complexity'])
    dycomp = [db_to_amp(dc) for dc in dycomp]
    
    plot_lowesses(dates, [loudness, dycomp, lvarcoeff],
        ['loudness', 'dycomp', 'loudness var'], PATH+'*overall.lowess2.dynamics.png')

def db_to_amp(db):
    return 10**(np.array(db)/20)

def plot_pitch():
    print('essentias')
    dates, essentias = get_essentias()
    
    print('pitch/tuning')
    hpcp = [get_essentia_hists(e, ['tonal','hpcp','mean']) for e in essentias]
    chroma = [[np.mean(np.reshape(np.roll(h, 1), (-1,3)), axis=1)
        for h in hs] for hs in hpcp]
    pitchvar = [[pdf_freq_variation(c) for c in cs] for cs in chroma]
    
    #tuning deviation
    tuning = [[np.mean(np.reshape(np.roll(h, 1), (-1,3)), axis=0)
        for h in hs] for hs in hpcp]
    tuningvar = [[pdf_freq_variation(t) for t in ts] for ts in tuning]
    
    #tuning deviation +/-
    tuning2 = [[(t[2]-t[0])/np.sum(t) for t in ts] for ts in tuning]
    
    print('tonal complexity')
    tonalcomp = [util.multiprocess('tc', features.tonal_complexity_cf2, cs) for cs in chroma]
    tonalcomp = [[t**5 for t in ts] for ts in tonalcomp]
    
    print('lowesses')
    ax = plot_lowesses(dates, [pitchvar, tuningvar, tonalcomp],
        ['pitch var', 'tuning var', 'tonal comp'], relative=[True, True, True],
        path=PATH+'*overall.lowess2.pitch.png')
    
    # ax.legend(loc='upper left')
    # years = np.concatenate(dates).astype('datetime64[Y]')
    # counts = np.unique(years.astype(int), axis=0, return_counts=True)[1]
    # ax = ax.twinx()
    # ax.plot(np.unique(years), lowess(np.unique(years), counts/np.max(counts)),
    #     label='versions', color='green')
    # ax.legend()
    # plot(PATH+'*overall.lowess2.pitch-abs.png')

def plot_spectral():
    dates, essentias = get_essentias()
    
    complexity = get_essentia(essentias, ['lowlevel','spectral_complexity','median'])
    complexity = [freq_to_linear(c) for c in complexity]
    centroid = get_essentia(essentias, ['lowlevel','spectral_centroid','median'])
    centroid = [freq_to_linear(c) for c in centroid]
    entropy = get_essentia(essentias, ['lowlevel','spectral_entropy','median'])
    dissonance = get_essentia(essentias, ['lowlevel','dissonance','median'])
    estd = get_essentia(essentias, ['lowlevel','spectral_entropy','stdev'])
    emean = get_essentia(essentias, ['lowlevel','spectral_entropy','mean'])
    evarcoeff = [db_to_amp(s)/db_to_amp(m) for s,m in zip(estd, emean)]
    
    # mfcc = get_essentia(essentias, ['lowlevel','mfcc','mean'])[2:]
    # mfccvar = [[pdf_freq_variation(m) for m in mf] for mf in mfcc]
    
    # speccompvars, rspeccompvars = get_essentia_relative_varcoeffs(pdates, essentias,
    #     ['lowlevel','spectral_complexity'])[1:]
    plot_lowesses(dates, [complexity, centroid, entropy, dissonance, evarcoeff],
        ['complexity', 'centroid', 'entropy', 'dissonance', 'entropy var'],
        PATH+'*overall.lowess2.spectral.png')

def freq_to_linear(freq):
    return np.log2(freq)

def plot_absolute_relative_comparison():
    songs = gd.SONGS[:]
    chords, beats, dates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    tempos = [tempo(b) for b in beats]
    durations = [[b[-1]-b[0] for b in bs] for bs in beats]
    # essentias = get_essentias()[1]
    # complexity = get_essentia(essentias, ['lowlevel','spectral_complexity','mean'])
    # hpcp = [get_essentia_hists(e, ['tonal','hpcp','mean']) for e in essentias]
    # chroma = [[np.mean(np.reshape(np.roll(h, 1), (-1,3)), axis=1)
    #     for h in hs] for hs in hpcp]
    # pitchvar = [[pdf_freq_variation(c) for c in cs] for cs in chroma]
    # chordvar = [[freq_variation(c) for c in cs] for cs in chords]
    
    plot_lowesses(dates, [tempos, tempos, durations, durations],
        ['tempo rel', 'tempo abs', 'duration rel', 'duration abs'],
        PATH+'*overall.lowess2.absrel.png', [True, False, True, False])

def get_essentias():
    songs = gd.SONGS[:]
    chords, beats, dates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    essentias = [gd.get_essentias(s) for s in songs]
    essentias = [[e for i,e in enumerate(es) if i not in r]
        for es,r in zip(essentias, removed)]
    return dates, essentias

def to_patterns(features):
    median = np.median(np.concatenate(np.concatenate(features)), axis=0)#np.quantile(np.concatenate(np.concatenate(features)), 0.25, axis=0)
    return [[(c >= median).astype(int) for c in cs] for cs in features]

def plot_lowesses(dates, features, names, path=None, relative=None):
    if relative is None: relative = np.ones(len(features)) #default relative True
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,(f,n,r) in enumerate(zip(features, names, relative)):
        plot_lowess_bootstrap(ax, dates, f, n, colors[i], r)
    if path:
        plt.legend()
        plot(path)
    return ax

def plot_lowess_bootstrap(ax, pdates, pvalues, name, color, relative):
    dates, values, lowess, mins, maxes = get_lowess_bootstrap(pdates, pvalues, relative)
    if not relative:#normalize for comparison
        lowess, mins, maxes = lowess/np.mean(lowess), mins/np.mean(lowess), maxes/np.mean(lowess)
        #ax = ax.twinx()
    ax.fill_between(dates, mins, maxes, alpha=0.2, color=color)
    #plt.plot(dates, util.running_mean(rvalues, 100), color=color, label=name)
    plt.plot(dates, lowess, color=color, label=name)
    #plt.plot(dates, lowess2(dates, rvalues), linewidth=2, label=name+' lowess2')

def get_lowess_bootstrap(pdates, pvalues, relative):
    alldates = np.unique(np.concatenate(pdates))
    maxes = {}
    mins = {}
    for i in range(len(pdates)):
        dates = [d for j,d in enumerate(pdates) if j != i]
        values = [t for j,t in enumerate(pvalues) if j != i]
        dates, values, rvalues = merge_with_relative(dates, values)
        values = rvalues if relative else values
        #dates, rvalues = average_same_dates(dates, rvalues)
        values = lowess(dates, values, alldates)
        for d,t in zip(alldates, values):
            maxes[d] = max(t, maxes[d]) if d in maxes else t
            mins[d] = min(t, mins[d]) if d in mins else t
        #plt.plot(alldates, values, linewidth=1, label=name+' w/o '+str(i))
    maxes = [maxes[d] for d in alldates]
    mins = [mins[d] for d in alldates]
    
    dates, values, rvalues = merge_with_relative(pdates, pvalues)
    values = rvalues if relative else values
    #dates, values = average_same_dates(dates, values)
    
    return alldates, values, lowess(dates, values, alldates), mins, maxes

def lowess(dates, values, alldates=None, frac=0.2):
    to_timestamps = lambda d: np.array(d, dtype='datetime64[s]').astype('int')
    dates = to_timestamps(dates)
    alldates = to_timestamps(alldates) if alldates is not None else dates
    return sm.nonparametric.lowess(values, dates, frac=frac, xvals=alldates, is_sorted=True)

def lowess2(dates, values):
    timestamps = dates.astype('datetime64[s]').astype('int')
    l = loess(timestamps, values)
    l.fit()
    pred = l.predict(timestamps, stderror=True)
    return pred.values

def average_same_dates(dates, values):
    uniq_dates = np.unique(dates)
    return uniq_dates, [np.mean(values[np.where(dates == d)]) for d in uniq_dates]

def plot_tempo_combi():
    songs = gd.SONGS
    pchords, pbeats, pdates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    
    ptempos = [tempo(b) for b in pbeats]
    # plot_with_relative(pdates, tempos, '*overall_tempo')
    dates, tempos, rtempos = merge_with_relative(pdates, ptempos)[:]
    
    for i in range(len(songs)):
        #plt.plot(pdates[i], util.running_mean(ptempos[i], 20), linewidth=1, label=songs[i])
        plt.plot(pdates[i], lowess(pdates[i], ptempos[i]), linewidth=2, label=songs[i])
    # plt.plot(dates, util.running_mean(tempos/np.mean(tempos), 300), linewidth=2, label='abs tempo')
    # plt.plot(dates, util.running_mean(rtempos, 300), linewidth=2, label='rel tempo')
    #plt.plot(dates, util.running_mean(tempos, 300), linewidth=2, label='abs tempo')
    #plt.plot(dates, util.running_mean(rtempos, 300)*np.mean(tempos), linewidth=2, label='rel tempo')
    plt.legend()
    plot(PATH+'*overall.tempo.png')

def plot_overall_evolution():
    # #original
    # versions, dates = list(zip(*[gd.get_versions_by_date(s) for s in gd.SONGS]))
    # beats = [gd.get_beats(s) for s in gd.SONGS]
    # dates, versions, beats = combine_songs(dates, versions, beats)
    # plot_with_mean(PATH+'*overall_tempo.png', dates, tempo(beats))
    songs = gd.SONGS
    #preprocessed
    pchords, pbeats, pdates, removed = list(zip(*[get_preprocessed(s) for s in songs]))
    
    tempos = [tempo(b) for b in pbeats]
    # plot_with_relative(pdates, tempos, '*overall_tempo')
    dates, tempos, rtempos = merge_with_relative(pdates, tempos)[:]
    
    durations = [[b[-1]-b[0] for b in bs] for bs in pbeats]
    # plot_with_relative(pdates, durations, '*overall_duration')
    durations, rdurations = merge_with_relative(pdates, durations)[1:]
    
    beatcounts = [[len(b) for b in bs] for bs in pbeats]
    beatcounts, rbeatcounts = merge_with_relative(pdates, beatcounts)[1:]
    
    beatdurs = [[b[1:]-b[:-1] for b in bs] for bs in pbeats]
    beatvars = [[np.std(b)/np.mean(b) for b in bs] for bs in beatdurs]
    beatvars, rbeatvars = merge_with_relative(pdates, beatvars)[1:]
    rbeatvars2 = beatvars/np.mean(beatvars)
    
    chordcounts = [[len(np.unique(c)) for c in cs] for cs in pchords]
    # plot_with_relative(pdates, chordcounts, '*overall_chordcount')
    chordcounts, rchordcounts = merge_with_relative(pdates, chordcounts)[1:]
    
    chordvars = [[freq_variation(c) for c in cs] for cs in pchords]
    # plot_with_relative(pdates, chordvars, '*overall_chordvar')
    chordvars, rchordvars = merge_with_relative(pdates, chordvars)[1:]
    
    chordents = [[entropy2(c) for c in cs] for cs in pchords]
    # plot_with_relative(pdates, chordents, '*overall_chordent')
    chordents, rchordents = merge_with_relative(pdates, chordents)[1:]
    
    #essentia stuff
    essentias = [gd.get_essentias(s) for s in songs]
    essentias = [[e for i,e in enumerate(es) if i not in r]
        for es,r in zip(essentias, removed)]
    #essentias = merge_by_dates(dates, essentias)[1]
    
    print('essentias')
    hpcp = [get_essentia_hists(e, ['tonal','hpcp','mean']) for e in essentias]
    chroma = [[np.mean(np.reshape(np.roll(h, 1), (-1,3)), axis=1)
        for h in hs] for hs in hpcp]
    chromavars = [[pdf_freq_variation(c) for c in cs] for cs in chroma]
    chromavars, rchromavars = merge_with_relative(pdates, chromavars)[1:]
    
    #tuning deviation
    tuning = [[np.mean(np.reshape(np.roll(h, 1), (-1,3)), axis=0)
        for h in hs] for hs in hpcp]
    tuningvars = [[pdf_freq_variation(t) for t in ts] for ts in tuning]
    tuningvars, rtuningvars = merge_with_relative(pdates, tuningvars)[1:]
    
    #tuning deviation +/-
    tuning2 = [[(t[2]-t[0])/np.sum(t) for t in ts] for ts in tuning]
    tuning2, rtuning2 = merge_with_relative(pdates, tuning2)[1:]
    
    loudnesses, rloudnesses = get_essentia_relative(pdates, essentias,
        ['lowlevel','loudness_ebu128','short_term','mean'])[1:]
    loudvars, rloudvars = get_essentia_relative_varcoeffs(pdates, essentias,
        ['lowlevel','loudness_ebu128','short_term'])[1:]
    dycomps, rdycomps = get_essentia_relative(pdates, essentias,
        ['lowlevel','dynamic_complexity'])[1:]
    
    speccomps, rspeccomps = get_essentia_relative(pdates, essentias,
        ['lowlevel','spectral_complexity','mean'])[1:]
    speccompvars, rspeccompvars = get_essentia_relative_varcoeffs(pdates, essentias,
        ['lowlevel','spectral_complexity'])[1:]
    speccent, rspeccent = get_essentia_relative(pdates, essentias,
        ['lowlevel','spectral_centroid','mean'])[1:]
    specent, rspecent = get_essentia_relative(pdates, essentias,
        ['lowlevel','spectral_entropy','mean'])[1:]
    dissonances, rdissonances = get_essentia_relative(pdates, essentias,
        ['lowlevel','dissonance','mean'])[1:]
    
    
    #plots
    window = 300
    
    # plt.plot(dates, util.running_mean(rchordcounts, window), label='chord counts')
    plt.plot(dates, util.running_mean(rchordvars, window), label='chord vars')
    plt.plot(dates, util.running_mean(rchordents, window), label='chord ents')
    plt.plot(dates, util.running_mean(rchromavars, window), label='pitch-class vars')
    plt.plot(dates, util.running_mean(rtuningvars, window), label='tuning vars')
    plt.plot(dates, util.running_mean(rtuning2, window), label='rel tuning')
    plt.legend()
    plot(PATH+'*overall.tonal.png')
    
    plt.plot(dates, util.running_mean(rloudnesses, window), label='loudness')
    plt.plot(dates, util.running_mean(rloudvars, window), label='loudness var')
    plt.plot(dates, util.running_mean(rdycomps, window), label='dynamic complexity')
    plt.legend()
    plot(PATH+'*overall.dynamics.png')
    
    plt.plot(dates, util.running_mean(rspeccomps, window), label='spec complexity')
    plt.plot(dates, util.running_mean(rspeccompvars, window), label='spec complexity var')
    plt.plot(dates, util.running_mean(rspeccent, window), label='spec centroid')
    plt.plot(dates, util.running_mean(rspecent, window), label='spec entropy')
    plt.plot(dates, util.running_mean(rdissonances, window), label='dissonance')
    plt.legend()
    plot(PATH+'*overall.spectral.png')
    
    plt.plot(dates, util.running_mean(rtempos, window), label='tempos')
    plt.plot(dates, util.running_mean(rdurations, window), label='durations')
    plt.plot(dates, util.running_mean(rbeatcounts, window), label='beat counts')
    #duration*rtempo proportional to beatcount
    #plt.plot(dates, util.running_mean(np.multiply(rdurations, rtempos), window), label='dur*tempo')
    plt.plot(dates, util.running_mean(rbeatvars, window), label='beat vars')
    #plt.plot(dates, util.running_mean(rbeatvars2, window), label='beat vars 2')
    plt.legend()
    plot(PATH+'*overall.time.png')
    
    #plot_essentias(PATH+'*overall', pdates, essentias, 100)

def plot_with_relative(dates, features, filename):
    dates, features, relative = overall_with_relative(dates, features)
    plot_with_mean(PATH+filename+'.png', dates, features, 100)
    plot_with_mean(PATH+filename+'_rel.png', dates, relative, 100)

def get_essentia(essentias, keys):
    return [[dict_value(e, keys) for e in es] for es in essentias]

def get_essentia_relative(dates, essentias, keys):
    return merge_with_relative(dates, get_essentia(essentias, keys))

def get_essentia_relative_varcoeffs(dates, essentias, keys):
    varcoeffs = [[dict_value(e, keys)['stdev'] / dict_value(e, keys)['mean']
        for e in es] for es in essentias]
    return merge_with_relative(dates, varcoeffs)

def merge_with_relative(dates, features_per_song):
    relative = [f/np.mean(f) for f in features_per_song]
    return merge_by_dates(dates, features_per_song, relative)

def merge_by_dates(dates, *features):
    dates = util.flatten(list(dates))
    features = [util.flatten(list(f), 1) for f in features]
    return [np.array(z) for z in
        zip(*sorted(list(zip(dates, *features)), key=lambda l: l[0]))]

def plot_individual_evolutions():
    for s in tqdm.tqdm(gd.SONGS, desc='tempos'):
        plot_evolution(s)

def plot_evolution(song):
    # oversions, odates = gd.get_versions_by_date(song)
    # obeats = gd.get_beats(song)
    chords, beats, dates, removed = get_preprocessed(song)
    # plot_with_mean(PATH+song+'_tempo.png', dates, tempo(beats))
    # durations = [b[-1]-b[0] for b in beats]
    # plot_with_mean(PATH+song+'_duration.png', dates, durations)
    # plot_with_mean(PATH+song+'_beats.png', dates, [len(b) for b in beats])
    
    # #tempo comparison
    # plt.plot(dates, util.running_mean(tempo(beats), 10))
    # plt.plot(pdates, util.running_mean(tempo(pbeats), 10))
    # plot(PATH+song+'_tempocomp')
    
    # plot_with_mean(PATH+song+'_chordvacos.png', dates,
    #     [freq_variation(c) for c in chords])
    # plot_with_mean(PATH+song+'_chordents.png', dates,
    #     [entropy2(c) for c in chords])
    
    essentias = [e for i,e in enumerate(gd.get_essentias(song)) if i not in removed]
    
    # plot_essentia_hist(essentias, ['tonal','hpcp','mean'], PATH+song+'_hpcp.png')
    # plot_essentia_hist(essentias, ['lowlevel','melbands','mean'], PATH+song+'_melbands.png')
    # plot_essentia_hist(essentias, ['lowlevel','mfcc','mean'], PATH+song+'_mfcc.png')
    # plot_essentia_hist(essentias, ['tonal','chords_histogram'], PATH+song+'_chordhists.png')
    
    #onsets = [np.array(e['rhythm']['onset_times']) for e in essentias]
    
    onsets = [o for i,o in enumerate(gd.get_onsets(song)) if i not in removed]
    # plot_onsethists(song, onsets, beats)
    # plot_onsethistpeaks(song, onsets, beats)
    plot_onsethistrealpeaks(song, onsets, beats)
    
    # onset_durs = [(o[1:]-o[:-1]) for i,o in enumerate(onsets)]
    # plot_with_mean(PATH+song+'_onsetvacos.png', dates,
    #     [np.std(o)/np.mean(o) for o in onset_durs])
    # beat_durs = [(b[1:]-b[:-1]) for i,b in enumerate(beats)]
    # plot_with_mean(PATH+song+'_beatvacos.png', dates,
    #     [np.std(b)/np.mean(b) for b in beat_durs])
    
    
    # plot_essentias(PATH+song, dates, essentias, 10)

def plot_onsethists(song, onsets, beats):
    hists = get_onset_hists(onsets,beats)
    util.plot_matrix(np.rot90(hists), PATH+song+'_onsethists3.png')

def plot_onsethistpeaks(song, onsets, beats):
    hists = get_onset_hists(onsets,beats)
    #print(hists[:3])
    peaks = [find_peaks(h, prominence=0.5, distance=10)[0] for h in hists]
    print(peaks[:3])
    matrix = np.zeros((len(hists), len(hists[0])))
    for i,p in enumerate(peaks):
        matrix[i,p] = 1
    util.plot_matrix(np.rot90(matrix), PATH+song+'_onsethists4.png')

def plot_onsethistrealpeaks(song, onsets, beats, resolution=1000):
    onsetpos = [get_onsetpos(o,b) for o,b in zip(onsets,beats)]
    densities = [stats.gaussian_kde(o) for o in onsetpos]
    values = [d.evaluate(np.linspace(o.min(), o.max(), resolution))
        for o,d in zip(onsetpos, densities)]
    #print(values[0])
    peaks = [find_peaks(v)[0] for v in values]
    print(peaks[:10])
    matrix = np.zeros((len(onsetpos), resolution))
    for i,p in enumerate(peaks):
        matrix[i,p] = 1
    util.plot_matrix(np.rot90(matrix), PATH+song+'_onsetsreal2.png')

def plot_essentia_hist(essentias, keys, path):
    util.plot_matrix(np.rot90(get_essentia_hists(essentias, keys)), path)

def get_essentia_hists(essentias, keys):
    return [dict_value(e, keys) for e in essentias]

def dict_value(dict, keys):
    return reduce(lambda a,b: a.get(b,{}), keys, dict)

def plot_essentias(path, dates, essentias, window):
    ess = [e['lowlevel']['dynamic_complexity'] for e in essentias]
    plot_with_mean(path+'_dycomp.png', dates, ess, window)
    
    ess = [e['lowlevel']['spectral_complexity']['mean'] for e in essentias]
    plot_with_mean(path+'_speccomp.png', dates, ess, window)
    
    ess = [e['lowlevel']['spectral_entropy']['mean'] for e in essentias]
    plot_with_mean(path+'_specent.png', dates, ess, window)
    
    ess = [e['lowlevel']['spectral_flux']['mean'] for e in essentias]
    plot_with_mean(path+'_speccomp.png', dates, ess, window)
    
    ess = [e['tonal']['chords_changes_rate'] for e in essentias]
    plot_with_mean(path+'_chordcha.png', dates, ess, window)

def get_preprocessed(song):
    chords, beats = main.get_preprocessed_seqs(song)
    _, dates = gd.get_versions_by_date(song)
    removed = [i for i,c in enumerate(chords) if c is None]
    chords = [c for c in chords if c is not None]
    beats = [b for b in beats if b is not None]
    dates = [d for i,d in enumerate(dates) if i not in removed]
    return chords, beats, dates, removed

def plot_with_mean(path, dates, feature, window=10):
    plt.plot(dates, feature)
    plt.plot(dates, util.running_mean(feature, window))
    plot(path)

def pdf_freq_variation(feature_pdf): #std = mean for exp dist; TODO think about best measure
    feature_pdf /= np.sum(feature_pdf) #normalize just in case
    return pdf_var(sorted(feature_pdf, reverse=True))

#returns the variation coefficient of a qualitative feature
def freq_variation(feature):
    return pdf_var(freq_pdf(feature))

def pdf_var(pdf): #std = mean for exp dist; TODO think about best measure?
    #return np.std(dist)/np.mean(dist) #coefficient of variation inappropriate..
    return np.sum(np.arange(len(pdf)) * pdf)

#returns the sorted distribution of frequencies of a qualitative feature
def freq_pdf(feature):
    hist = sorted(np.unique(feature, axis=0, return_counts=True)[1], reverse=True)
    return hist/np.sum(hist)

def plot(path=None):
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.close()

# def plot_evolution(song):
#     v, d = gd.get_versions_by_date(song)
# 
#     #b = [e['rhythm']['onset_rate'] for e in get_essentias(SONGS[SONG_INDEX])]
#     b = [e['lowlevel']['dynamic_complexity'] for e in gd.get_essentias(song)]
#     #b = [e['lowlevel']['dynamic_complexity'] for e in gd.get_essentias(song)]
#     #b = [e['metadata']['audio_properties']['length'] for e in get_essentias(SONGS[SONG_INDEX])]
#     #b = [b/2 if b > 140 else b for b in b]
# 
#     beats = gd.get_beats(song)
#     chords = gd.get_chord_sequences(song)
#     print(chords[0][:10])
#     print([len(c) for c in chords[:]])
#     chords = main.get_preprocessed_seqs(song)
#     print(chords[0][:10])
#     print([len(c) for c in chords[:]])
# 
#     tempos = 
# 
# 
#     # TEST HALF/DOUBLE TIME...
#     # AND CLASSIFICATION...
# 
#     print(len(b))
#     plt.plot(d, b)
#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0)) 
#         return (cumsum[N:] - cumsum[:-N]) / float(N)
#     b = running_mean(b, 10)
#     print(len(b))
#     b = np.pad(b, (5,4), 'constant', constant_values=(0,0))
#     plt.plot(d, b)
#     plt.show()

if __name__ == "__main__":
    #gd.extract_onsets_for_all()
    #gd.extract_beats_for_all()
    #gd.extract_mfcc_for_all()
    #plot_overall_evolution()
    #plot_individual_evolutions()
    #plot_tempo_combi()
    plot_all()