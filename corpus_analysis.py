import os, json, timeit
import numpy as np
from features import get_beatwise_chords, to_multinomial
from alignment import get_alignment_segments, get_affinity_matrix, get_alignment_matrix
from multi_alignment import align_sequences
from util import profile, plot_matrix, buffered_run

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
    versions = list(dataset[song].keys())
    paths = [get_feature_path(song, v) for v in versions]
    return [get_beatwise_chords(p+'_madbars.json', p+'_gochords.json')
        for p in paths]

def plot_matrices(sequence, max_gaps=0):
    matrix = get_affinity_matrix(sequence, sequence, True, max_gaps)
    matrix2 = get_alignment_matrix(sequence, sequence, 16, 4, max_gaps)
    plot_matrix(matrix, 'results/aff'+str(max_gaps))
    plot_matrix(matrix2, 'results/ali'+str(max_gaps))

def get_self_alignments(sequences, max_gaps=0):
    return [get_alignment_segments(s, s, 16, 4, max_gaps) for s in sequences]

def run(song):
    print('seqs')
    sequences = buffered_run('data/'+song+'-chords.npy',
        lambda: get_sequences(song))
    print('self')
    sas = buffered_run('data/'+song+'-salign.npy',
        lambda: get_self_alignments(sequences))
    print('multi')
    multinomial = buffered_run('data/'+song+'-mulnom.npy',
        lambda: to_multinomial(sequences))
    msa = buffered_run('data/'+song+'-msa.npy',
        lambda: align_sequences(multinomial)[0])
    print(msa)
    #profile(lambda: get_alignment(chords, chords, 16, 4, 0))
    #print(timeit.timeit(lambda: get_alignment(chords, chords, 16, 4, 0), number=1))

run(songs[0])