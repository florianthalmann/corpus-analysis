import os, json, timeit
import numpy as np
from multiprocessing import Pool
from features import get_beatwise_chords, to_multinomial, extract_essentia
from alignments import get_alignment_segments, get_affinity_matrix,\
    get_alignment_matrix, segments_to_matrix
from multi_alignment import align_sequences
from graphs import to_alignment_graph, get_component_labels, to_matrix
from util import profile, plot_matrix, buffered_run
from graph_tool.topology import transitive_closure
from hierarchies import make_hierarchical

corpus = '../../FAST/fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
with open(os.path.join(corpus, 'dataset.json')) as f:
    dataset = json.load(f)

def get_subdirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]

songs = list(dataset.keys())

def extract_features_for_song(song):
    versions = list(dataset[song].keys())
    audio_paths = [os.path.join(audio, song, v).replace('.mp3','.wav') for v in versions]
    feature_paths = [get_feature_path(song, v)+'_freesound.json' for v in versions]
    [extract_essentia(a, p) for (a, p) in zip(audio_paths, feature_paths)]

def extract_features():
    with Pool(processes=6) as pool:
        pool.map(extract_features_for_song, songs)

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
    sequences = buffered_run('data/'+song+'-chords.npy',
        lambda: get_sequences(song))
    sas = buffered_run('data/'+song+'-salign.npy',
        lambda: get_self_alignments(sequences, 1))
    multinomial = buffered_run('data/'+song+'-mulnom.npy',
        lambda: to_multinomial(sequences))
    msa = buffered_run('data/'+song+'-msa.npy',
        lambda: align_sequences(multinomial)[0])
    #g, s, i = to_alignment_graph([len(s) for s in sequences], sas)
    TEST_INDEX = 60
    #g, s, i = to_alignment_graph([len(sequences[TEST_INDEX])], [sas[TEST_INDEX]])
    #c = get_component_labels(g)
    #plot_matrix(to_matrix(g), 'results/oufuku1.png')
    #plot_matrix(to_matrix(transitive_closure(g)), 'results/oufuku2.png')
    size = len(sequences[TEST_INDEX])
    plot_matrix(segments_to_matrix(make_hierarchical(sas[TEST_INDEX], 16, 4),
        (size,size)), 'results/oufuku3.png')
    
    #profile(lambda: to_alignment_graph([len(s) for s in sequences], sas))
    #profile(lambda: get_alignment(chords, chords, 16, 4, 0))
    #print(timeit.timeit(lambda: get_alignment(chords, chords, 16, 4, 0), number=1))

run(songs[0])
