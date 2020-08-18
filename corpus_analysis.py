import os, json, timeit
import numpy as np
from features import get_beatwise_chords
from alignment import get_alignment_segments, get_affinity_matrix, get_alignment_matrix
from util import profile, plot_matrix

corpus = '../../FAST/fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
with open(os.path.join(corpus, 'dataset.json')) as f:
    dataset = json.load(f)

def get_subdirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]

songs = list(dataset.keys())

def get_feature_path(song, version):
    id = version.replace('.mp3','.wav').replace('.','_').replace('/','_')
    return os.path.join(features, id, id)

def get_sequences(song):
    print("loading")
    version = list(dataset[song].keys())[60]
    beatsFile = get_feature_path(song, version) + '_madbars.json'
    chordsFile = get_feature_path(song, version) + '_gochords.json'
    chords = np.array(get_beatwise_chords(beatsFile, chordsFile))
    
    max_gaps = 0
    matrix = get_affinity_matrix(chords, chords, True, max_gaps)
    matrix2 = get_alignment_matrix(chords, chords, 16, 4, max_gaps)
    segments = get_alignment_segments(chords, chords, 16, 4, max_gaps)
    
    #profile(lambda: get_alignment(chords, chords, 16, 4, 0))
    #print(timeit.timeit(lambda: get_alignment(chords, chords, 16, 4, 0), number=1))
    plot_matrix(matrix, 'aff'+str(max_gaps))
    plot_matrix(matrix2, 'ali'+str(max_gaps))

get_sequences(songs[0])