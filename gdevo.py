import tqdm
import numpy as np
from matplotlib import pyplot as plt
import gd, main
from corpus_analysis import util
from corpus_analysis.stats.util import entropy2

PATH='results/gdevo/'

def plot_overall_evolution():
    # #original
    # versions, dates = list(zip(*[gd.get_versions_by_date(s) for s in gd.SONGS]))
    # beats = [gd.get_beats(s) for s in gd.SONGS]
    # dates, versions, beats = combine_songs(dates, versions, beats)
    # plot_with_mean(PATH+'*overall_tempo.png', dates, tempo(beats))
    #preprocessed
    pchords, pbeats, pdates, removed = list(zip(*[get_preprocessed(s) for s in gd.SONGS]))
    essentias = [gd.get_essentias(s) for s in gd.SONGS]
    essentias = [[e for i,e in enumerate(es) if i not in r]
        for es,r in zip(essentias,removed)]
    pdates, pchords, pbeats, essentias = combine_songs(pdates, pchords, pbeats, essentias)
    plot_with_mean(PATH+'*overall_tempo.png', pdates, tempo(pbeats), 100)
    
    plot_essentias(PATH+'*overall', pdates, essentias, 100)

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
    
    # chordfreqs = [frequencies(c) for c in chords]
    # plot_with_mean(PATH+song+'_chordvacos.png', dates,
    #     [np.std(c)/np.mean(c) for c in chordfreqs])
    # plot_with_mean(PATH+song+'_chordents.png', dates,
    #     [entropy2(c) for c in chords])
    
    # onsets = [np.array(e['rhythm']['onset_times']) for e in gd.get_essentias(song)]
    # onsets = [o for i,o in enumerate(onsets) if i not in removed]
    # onset_durs = [(o[1:]-o[:-1]) for i,o in enumerate(onsets)]
    # plot_with_mean(PATH+song+'_onsetvacos.png', dates,
    #     [np.std(o)/np.mean(o) for o in onset_durs])
    # beat_durs = [(b[1:]-b[:-1]) for i,b in enumerate(beats)]
    # plot_with_mean(PATH+song+'_beatvacos.png', dates,
    #     [np.std(b)/np.mean(b) for b in beat_durs])
    
    essentias = [e for i,e in enumerate(gd.get_essentias(song)) if i not in removed]
    plot_essentias(PATH+song, dates, essentias, 10)

def plot_essentias(path, dates, essentias, window):
    ess = [e['lowlevel']['dynamic_complexity'] for e in essentias]
    plot_with_mean(path+'_dycomp.png', dates, ess, 100)
    
    ess = [e['lowlevel']['spectral_complexity']['mean'] for e in essentias]
    plot_with_mean(path+'_speccomp.png', dates, ess, 100)
    
    ess = [e['lowlevel']['spectral_entropy']['mean'] for e in essentias]
    plot_with_mean(path+'_specent.png', dates, ess, 100)
    
    ess = [e['lowlevel']['spectral_flux']['mean'] for e in essentias]
    plot_with_mean(path+'_speccomp.png', dates, ess, 100)
    
    ess = [e['tonal']['chords_changes_rate'] for e in essentias]
    plot_with_mean(path+'_chordcha.png', dates, ess, 100)

def combine_songs(dates, *features):
    dates = util.flatten(list(dates))
    features = [util.flatten(list(f), 1) for f in features]
    return zip(*sorted(list(zip(dates, *features)), key=lambda l: l[0]))

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

def tempo(beats):
    return [60/np.mean(b[1:]-b[:-1]) for b in beats]

def frequencies(feature):
    return sorted(np.unique(feature, return_counts=True)[1], reverse=True)

def plot(path):
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
    #plot_overall_evolution()
    plot_individual_evolutions()