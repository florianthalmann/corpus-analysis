import os, json, dateutil, datetime
from corpus_analysis.features import get_summarized_chords, to_multinomial, extract_essentia,\
    load_leadsheets, get_summarized_chords2, get_beat_summary, get_summarized_chroma,\
    get_summarized_mfcc, extract_chords, load_essentia

corpus = '../fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
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

def get_paths(song):
    versions = get_versions_by_date(song)[0]
    audio_paths = [os.path.join(audio, song, v).replace('.mp3','.wav') for v in versions]
    feature_paths = [get_feature_path(song, v)+'_freesound.json' for v in versions]
    return audio_paths, feature_paths

def get_essentias(song):
    return [load_essentia(p) for p in get_paths(song)[1]]

def extract_features_for_song(song):
    audio_paths, feature_paths = get_paths(song)
    [extract_essentia(a, p) for (a, p) in zip(audio_paths, feature_paths)]

def get_feature_path(song, version):
    id = version.replace('.mp3','.wav').replace('.','_').replace('/','_')
    return os.path.join(features, id, id)

def get_chroma_sequences(song):
    audio = get_paths(song)[0]
    versions = get_versions_by_date(song)[0]
    paths = [get_feature_path(song, v) for v in versions]
    return [get_summarized_chroma(audio[i], p+'_madbars.json')
        for i,p in enumerate(paths)]

def get_chord_sequences(song):
    versions = get_versions_by_date(song)[0]
    paths = [get_feature_path(song, v) for v in versions]
    return [get_summarized_chords(p+'_madbars.json', p+'_gochords.json', BARS)
        for p in paths]