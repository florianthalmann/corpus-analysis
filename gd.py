import os, json, dateutil, datetime, tqdm
from corpus_analysis.features import get_summarized_chords, to_multinomial, extract_essentia,\
    load_leadsheets, get_summarized_chords2, get_beat_summary, get_summarized_chroma,\
    get_summarized_mfcc, extract_chords, load_essentia, load_beats, extract_onsets,\
    load_onsets, extract_beats
from corpus_analysis.util import multiprocess

corpus = os.path.abspath('/Users/flo/Projects/Code/Kyoto/fifteen-songs-dataset2/')
audio = os.path.abspath('/Users/flo/Desktop/migration/tuned_audio/')#os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
leadsheets = os.path.join(corpus, 'leadsheets')
with open(os.path.join(corpus, 'dataset.json')) as f:
    DATASET = json.load(f)

SONGS = list(DATASET.keys())

def get_versions_by_date(song):
    versions = list(DATASET[song].keys())
    dates = [dateutil.parser.parse(p.split('.')[0][2:]) for p in versions]
    dates = [d.replace(year=d.year-100) if d > datetime.datetime.now() else d
        for d in dates]
    versions, dates = zip(*sorted(zip(versions, dates), key=lambda vd: vd[1]))
    return list(versions), list(dates)

def get_paths(song, feature_ext=''):
    versions = get_versions_by_date(song)[0]
    audio_paths = [os.path.join(audio, song, v).replace('.mp3','.wav') for v in versions]
    feature_paths = [get_feature_path(song, v)+feature_ext for v in versions]
    return audio_paths, feature_paths

def get_feature_path(song, version):
    id = version.replace('.mp3','.wav').replace('.','_').replace('/','_')
    return os.path.join(features, id, id)

def get_essentias(song):
    return [load_essentia(p) for p in get_paths(song, '_freesound.json')[1]]

def extract_essentia_for_song(song):
    [extract_essentia(a, p) for a,p in zip(*get_paths(song, '_freesound.json'))]

#onsets: CNNOnsetDetector, onsets2: OnsetDetector
def extract_onsets_for_all():
    for s in SONGS:
        multiprocess('onsets '+s, extract_onsets,
            list(zip(*get_paths(s, '_onsets.txt'))))

def extract_beats_for_all():
    for s in SONGS:
        multiprocess('beats '+s, extract_beats,
            list(zip(*get_paths(s, '_beats.txt'))))

def get_feature_paths(song):
    versions = get_versions_by_date(song)[0]
    return [get_feature_path(song, v) for v in versions]

BEATS_EXT = '_beats.txt'#'_madbars.json'

def get_chroma_sequences(song):
    audio = get_paths(song)[0]
    return [get_summarized_chroma(audio[i], p+BEATS_EXT)
        for i,p in enumerate(get_feature_paths(song))]

def get_chord_sequences(song):
    return [get_summarized_chords(p+BEATS_EXT, p+'_gochords.json')
        for p in get_feature_paths(song)]

def get_beats(song):
    return [load_beats(p+BEATS_EXT) for p in get_feature_paths(song)]

def get_onsets(song):
    return [load_onsets(p+'_onsets.txt') for p in get_feature_paths(song)]