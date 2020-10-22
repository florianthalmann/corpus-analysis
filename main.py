import os, json, timeit, random
import numpy as np
from multiprocessing import Pool
from mir_eval.hierarchy import lmeasure
from corpus_analysis.features import get_summarized_chords, to_multinomial, extract_essentia,\
    load_leadsheets
from corpus_analysis.alignment.affinity import get_alignment_segments, get_affinity_matrix,\
    get_alignment_matrix, segments_to_matrix
from corpus_analysis.alignment.multi_alignment import align_sequences
from corpus_analysis.alignment.smith_waterman import smith_waterman
from corpus_analysis.util import profile, plot_matrix, plot_hist, plot, buffered_run
from corpus_analysis.hcomparison import get_relative_meet_triples, get_meet_matrix
from corpus_analysis.structure.structure import shared_structure, simple_structure,\
    illustrate_transitivity
from corpus_analysis.structure.laplacian import laplacian_segmentation

corpus = '../../FAST/fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
leadsheets = os.path.join(corpus, 'leadsheets')
with open(os.path.join(corpus, 'dataset.json')) as f:
    dataset = json.load(f)

DATA = 'data/'
BARS = False
#alignment
MIN_LEN = 16
MIN_DIST = 1 # >= 1
MAX_GAPS = 4
MAX_GAP_RATIO = .2
NUM_MUTUAL = 0

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
    return [get_summarized_chords(p+'_madbars.json', p+'_gochords.json', BARS)
        for p in paths]

def plot_matrices(sequence):
    matrix = get_affinity_matrix(sequence, sequence, True, MAX_GAPS)
    matrix2 = get_alignment_matrix(sequence, sequence, MIN_LEN, MIN_DIST, MAX_GAPS)
    plot_matrix(matrix, 'results/aff'+str(MAX_GAPS))
    plot_matrix(matrix2, 'results/ali'+str(MAX_GAPS))

def get_self_alignments(sequences):
    return [get_alignment_segments(s, s, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)
        for s in sequences]

def get_random_pairings(length):
    perm = np.random.permutation(np.arange(length))
    #add another pairing for the odd one out
    if length % 2 == 1: perm = np.append(perm, random.randint(0, length-1))
    return perm.reshape((2,-1)).T

def get_pairings(sequences):
    return np.concatenate([get_random_pairings(len(sequences))
        for i in range(NUM_MUTUAL)])

def get_mutual_alignments(sequences, pairings):
    return [get_alignment_segments(sequences[p[0]], sequences[p[1]],
        MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO) for p in pairings]

def plot_hists(alignment):
    points = [p for s in alignment
        for p in [s[0][0], s[0][1], s[-1][0], s[-1][1]]]
    plot_hist(points, 'results/hist_points4.png')
    lengths = [len(s) for s in alignment]
    plot_hist(lengths, 'results/hist_lengths4.png')
    dias = [s[0][1]-s[0][0] for s in alignment]
    plot_hist(dias, 'results/hist_dias4.png')

def get_alignments(song):
    sequences = buffered_run(DATA+song+'-chords.npy',
        lambda: get_sequences(song))
    selfs = buffered_run(DATA+song+'-salign'+str(MAX_GAPS)+'_'+str(MAX_GAP_RATIO)
        +'_'+str(MIN_LEN)+'_'+str(MIN_DIST)+'.npy',
        lambda: get_self_alignments(sequences))
    if NUM_MUTUAL > 0:
        pairings = buffered_run(DATA+song+'-pairs'+str(NUM_MUTUAL)+'.npy',
            lambda: get_pairings(sequences, NUM_MUTUAL))
        mutuals = buffered_run(DATA+song+'-malign'+str(MAX_GAPS)
                +'_'+str(MAX_GAP_RATIO)+'_'+str(MIN_LEN)
                +'_'+str(MIN_DIST)+'_'+str(NUM_MUTUAL)+'.npy',
            lambda: get_mutual_alignments(sequences, pairings))
    msa = buffered_run(DATA+song+'-msa.npy',
        lambda: align_sequences(sequences)[0])
    selfp = np.stack((np.arange(len(sequences)), np.arange(len(sequences)))).T
    pairings = np.concatenate((selfp, pairings)) if NUM_MUTUAL > 0 else selfp
    alignments = np.concatenate((selfs, mutuals)) if NUM_MUTUAL > 0 else selfs
    return sequences, pairings, alignments, msa

def get_intervals(levels):
    ivls = lambda l: np.stack([np.arange(l), np.arange(l)+1]).T
    return [ivls(len(l)) for l in levels]

def evaluate_hierarchy(reference, estimate):
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    #plot_matrix(segments_to_matrix([sw], (400,400)))
    print('smith waterman', len(sw))
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    print('raw lmeasure', lm)
    #rethink this multiplication...
    return lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]

def run():
    song_index = 0
    sequences, pairings, alignments, msa = get_alignments(songs[0])
    #plot_matrix(segments_to_matrix(mutuals[0]))
    #shared_structure(sequences, pairings, alignments, msa)
    #profile(lambda: shared_structure(sequences, sas, multinomial, msa))
    groundtruth = load_leadsheets(leadsheets, songs[:4])
    print("ground", groundtruth[song_index])
    #plot_matrix(get_alignment_matrix(sequences[62], sequences[62],
    #    MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO), 's4.png')
    #print("done")
    
    I1, I2 = 56, 62
    
    #print(sequences[I1][:100], groundtruth[song_index][3][:100])
    # plot_matrix(segments_to_matrix(
    #     [smith_waterman(sequences[I1], groundtruth[song_index][3])],
    #     (len(sequences[I1]), len(groundtruth[song_index][3]))))
    
    structure = simple_structure(sequences[I1], alignments[I1])
    #print(structure[:,:30])
    
    #print(sequences[I1])
    #print(structure[-1,:])
    #plot_matrix(smith_waterman(groundtruth[0][-1,:], sequences[I1])[1], 'sw1.png')
    #plot_matrix(smith_waterman(groundtruth[0][-1,:], structure[-1,:])[1], 'sw2.png')
    
    am = get_affinity_matrix(sequences[I1], sequences[I1], True, 0, 0)[0]
    laplacian = laplacian_segmentation(am)
    laplacian = np.append(laplacian, [sequences[I1]], axis=0)
    
    #print('original', len(np.array(smith_waterman(groundtruth[0][-1,:], sequences[I1])[0])))
    #print('improved', len(np.array(smith_waterman(groundtruth[0][-1,:], structure[-1,:])[0])))
    
    #plot_matrix(groundtruth[0])
    #plot_matrix(structure)
    
    # print("laplacian", evaluate_hierarchy(groundtruth[0], laplacian))
    # print("improved", evaluate_hierarchy(groundtruth[0], structure))
    # structure[-1] = sequences[I1]
    # print("original", evaluate_hierarchy(groundtruth[0], structure))
    
    #print(evaluate_hierarchy(groundtruth[1], structure))
    
    #evaluate_hierarchy(groundtruth[0][:,:400], groundtruth[1][:-1,:400])
    
    # print(lmeasure(get_intervals(groundtruth[0][:,:100]), groundtruth[0][:,:100],
    #     get_intervals(groundtruth[1][:,:100]), groundtruth[1][:,:100]))
    # 
    # print(lmeasure(get_intervals(groundtruth[0][:,:100]), groundtruth[0][:,:100],
    #     get_intervals(groundtruth[0][:,:100]), groundtruth[0][:,:100]))
    
    #illustrate_transitivity(multinomial[I1], alignments[I1])
    
    # plot_matrix(segments_to_matrix(alignments[I1]))
    # plot_matrix(segments_to_matrix(alignments[I2]))
    # #plot_hists(alignments[TEST_INDEX])
    # h1 = simple_structure(multinomial[I1], alignments[I1])
    # h2 = simple_structure(multinomial[I2], alignments[I2])
    # print(h1)
    # print(h2)
    #matrix = get_relative_meet_triples(hierarchy)
    #matrix = get_meet_matrix(hierarchy)
    #plot_matrix(matrix, 'results/meet_abs0_60-.png')

run()
