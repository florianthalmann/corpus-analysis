import os, json, timeit, random, itertools
import numpy as np
from mir_eval.hierarchy import lmeasure
from corpus_analysis.features import get_summarized_chords, to_multinomial, extract_essentia,\
    load_leadsheets, get_summarized_chords2
from corpus_analysis.alignment.affinity import get_alignment_segments, get_affinity_matrix,\
    get_alignment_matrix, segments_to_matrix
from corpus_analysis.alignment.multi_alignment import align_sequences
from corpus_analysis.alignment.smith_waterman import smith_waterman
from corpus_analysis.util import profile, plot_matrix, plot_hist, plot,\
    buffered_run, multiprocess, plot_graph
from corpus_analysis.structure.hcomparison import get_relative_meet_triples, get_meet_matrix
from corpus_analysis.structure.structure import shared_structure, simple_structure
from corpus_analysis.structure.laplacian import laplacian_segmentation, segment_file
from corpus_analysis.structure.graphs import graph_from_matrix, adjacency_matrix
import graph_tool

corpus = '../fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
leadsheets = os.path.join(corpus, 'leadsheets')
with open(os.path.join(corpus, 'dataset.json')) as f:
    DATASET = json.load(f)

DATA = 'data/'
BARS = False
#alignment
SEG_COUNT = 25
MIN_LEN = 16
MIN_DIST = 1 # >= 1
MAX_GAPS = 4
MAX_GAP_RATIO = .2
NUM_MUTUAL = 0
MIN_LEN2 = 20
MIN_DIST2 = 4
L_MAX_GAPS = 4
L_MAX_GAP_RATIO = .2

SONGS = list(DATASET.keys())

def get_paths(song):
    versions = list(DATASET[song].keys())
    audio_paths = [os.path.join(audio, song, v).replace('.mp3','.wav') for v in versions]
    feature_paths = [get_feature_path(song, v)+'_freesound.json' for v in versions]
    return audio_paths, feature_paths

def extract_features_for_song(song):
    audio_paths, feature_paths = get_paths(song)
    [extract_essentia(a, p) for (a, p) in zip(audio_paths, feature_paths)]

def extract_features():
    multiprocess('extracting features', extract_features_for_song, SONGS)

def get_feature_path(song, version):
    id = version.replace('.mp3','.wav').replace('.','_').replace('/','_')
    return os.path.join(features, id, id)

def get_sequences(song):
    versions = list(DATASET[song].keys())
    paths = [get_feature_path(song, v) for v in versions]
    return [get_summarized_chords(p+'_madbars.json', p+'_gochords.json', BARS)
        for p in paths]

def get_self_alignment(sequence):
    return get_alignment_segments(sequence, sequence, SEG_COUNT,
        MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)

def get_self_alignments(sequences):
    return multiprocess('self-alignments', get_self_alignment, sequences)

def get_random_pairings(length):
    perm = np.random.permutation(np.arange(length))
    #add another pairing for the odd one out
    if length % 2 == 1: perm = np.append(perm, random.randint(0, length-1))
    return perm.reshape((2,-1)).T

def get_pairings(sequences):
    return np.concatenate([get_random_pairings(len(sequences))
        for i in range(NUM_MUTUAL)])

def get_mutual_alignment(pairing_n_sequences):
    pairing, sequences = pairing_n_sequences
    return get_alignment_segments(sequences[pairing[0]], sequences[pairing[1]],
        SEG_COUNT, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)

def get_mutual_alignments(sequences, pairings):
    return multiprocess('mutual alignments', get_mutual_alignment,
        [(p, sequences) for p in pairings])

def get_alignments(song):
    sequences = buffered_run(DATA+song+'-chords',
        lambda: get_sequences(song))
    selfs = buffered_run(DATA+song+'-salign',
        lambda: get_self_alignments(sequences),
        [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST])
    if NUM_MUTUAL > 0:
        pairings = buffered_run(DATA+song+'-pairs',
            lambda: get_pairings(sequences), [NUM_MUTUAL])
        mutuals = buffered_run(DATA+song+'-malign',
            lambda: get_mutual_alignments(sequences, pairings),
            [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL])
    msa = buffered_run(DATA+song+'-msa',
        lambda: align_sequences(sequences)[0])
    selfp = np.stack((np.arange(len(sequences)), np.arange(len(sequences)))).T
    pairings = np.concatenate((selfp, pairings)) if NUM_MUTUAL > 0 else selfp
    alignments = np.concatenate((selfs, mutuals)) if NUM_MUTUAL > 0 else selfs
    return sequences, pairings, alignments, msa

def get_simple_structure(sequence_n_alignment):
    sequence, alignment = sequence_n_alignment
    return simple_structure(sequence, alignment, MIN_LEN2, MIN_DIST2)

def get_simple_structures(song, sequences, alignments):
    return buffered_run(DATA+song+'-structs',
        lambda: multiprocess('simple structures', get_simple_structure,
        list(zip(sequences, alignments))),
        [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL,
            MIN_LEN2, MIN_DIST2])

def get_laplacian_structure(sequence):
    affinity = get_affinity_matrix(sequence, sequence, True,
        L_MAX_GAPS, L_MAX_GAP_RATIO)[0]
    struct = laplacian_segmentation(affinity)
    return np.append(struct, [sequence], axis=0)

def get_laplacian_structures(song, sequences):
    return buffered_run(DATA+song+'-lstructs',
        lambda: multiprocess('laplacian structures', get_laplacian_structure,
        sequences), [L_MAX_GAPS, L_MAX_GAP_RATIO])

def get_orig_laplacian_struct(song_n_audio_n_version):
    song, audio, version = song_n_audio_n_version
    struct, beat_times = segment_file(audio)
    chords = get_summarized_chords2(beat_times,
        get_feature_path(song, version)+'_gochords.json')
    return np.append(struct, [chords], axis=0)

def get_orig_laplacian_structs(song):
    audio_files = get_paths(song)[0]
    versions = list(DATASET[song].keys())
    return buffered_run(DATA+song+'-olstructs',
        lambda: multiprocess('original laplacian structures', get_orig_laplacian_struct,
        list(zip(itertools.repeat(song), audio_files, versions))), [])

def get_intervals(levels):
    ivls = lambda l: np.stack([np.arange(l), np.arange(l)+1]).T
    return [ivls(len(l)) for l in levels]

def evaluate_hierarchy(reference_n_estimate):
    reference, estimate = reference_n_estimate
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    #plot_matrix(segments_to_matrix([sw], (400,400)))
    #print('smith waterman', len(sw))
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    #print('raw lmeasure', lm)
    #rethink this multiplication...
    return lm#lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]

def evaluate_hierarchies(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy,
        [(reference, e) for e in estimates]), [SEG_COUNT, MAX_GAPS,
            MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL, MIN_LEN2, MIN_DIST2])

def evaluate_hierarchies2(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy,
        [(reference, e) for e in estimates]), [L_MAX_GAPS, L_MAX_GAP_RATIO])

def evaluate_hierarchies3(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy,
        [(reference, e) for e in estimates]), [])

######################

def plot_matrices(sequence):
    matrix = get_affinity_matrix(sequence, sequence, True, MAX_GAPS)
    matrix2 = get_alignment_matrix(sequence, sequence, MIN_LEN, MIN_DIST, MAX_GAPS)
    plot_matrix(matrix, 'results/aff'+str(MAX_GAPS))
    plot_matrix(matrix2, 'results/ali'+str(MAX_GAPS))

def plot_hists(alignment):
    points = [p for s in alignment
        for p in [s[0][0], s[0][1], s[-1][0], s[-1][1]]]
    plot_hist(points, 'results/hist_points4.png')
    lengths = [len(s) for s in alignment]
    plot_hist(lengths, 'results/hist_lengths4.png')
    dias = [s[0][1]-s[0][0] for s in alignment]
    plot_hist(dias, 'results/hist_dias4.png')

def simplify_graph(alignment, length):
    g = graph_from_matrix(segments_to_matrix(alignment, (length, length)))
    print(g)
    g = graph_tool.all.Graph(g=g, directed=True)
    print(g)
    g = graph_tool.topology.transitive_closure(g)
    plot_matrix(adjacency_matrix(g), 'ztc.png')
    # plot_graph(g, 'z1g.png')
    # state = graph_tool.inference.minimize_blockmodel_dl(g)
    # state.draw(vertex_fill_color=state.b, output_size=(1000, 1000), bg_color=[1,1,1,1],
    #     output="z1bh.png")
    cliques = list(graph_tool.topology.max_cliques(g))
    cliques = [c for c in cliques if len(c) > 2]
    incli = g.new_vertex_property("bool")
    incli.a = np.isin(g.get_vertices(), np.concatenate(list(cliques)))
    cli = g.new_vertex_property("int")
    for i,c in enumerate(cliques):
        cli.a[c] = i
    print([c for c in cliques if 10 in c])
    print(sorted(cliques, key=lambda c: len(c), reverse=True)[:10])
    #g = graph_tool.all.GraphView(g, vfilt=incli)
    
    union = None
    for c in enumerate(cliques):
        incli.a = np.isin(g.get_vertices(), c)
        subgraph = graph_tool.all.GraphView(g, vfilt=incli)
        union = subgraph if not union else graph_tool.generation.graph_union(union, subgraph)
    
    graph_tool.all.graph_draw(union, vertex_fill_color=cli,
        output_size=(1000, 1000), bg_color=[1,1,1,1], output='z1c2.png')

def run():
    song_index = 0
    sequences, pairings, alignments, msa = get_alignments(SONGS[song_index])
    #print(len(np.concatenate(alignments[7])), len(np.concatenate(alignments[8])))
    print(sorted([len(a) for a in alignments]))
    #plot_matrix(segments_to_matrix(alignments[7]))
    # plot_matrix(segments_to_matrix(
    #     get_alignment_segments(sequences[7], sequences[7], SEG_COUNT, MIN_LEN, MIN_DIST, 4, .2),
    #     (len(sequences[7]), len(sequences[7]))))
    #shared_structure(sequences, pairings, alignments, msa)
    #profile(lambda: shared_structure(sequences, sas, multinomial, msa))
    groundtruth = load_leadsheets(leadsheets, SONGS)
    print("ground", groundtruth[song_index])
    #plot_matrix(get_alignment_matrix(sequences[62], sequences[62],
    #    MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO), 's4.png')
    #print("done")
    
    I1, I2 = 7, 62
    
    #print(sequences[I1][:100], groundtruth[song_index][3][:100])
    # plot_matrix(segments_to_matrix(
    #     [smith_waterman(sequences[I1], groundtruth[song_index][3])],
    #     (len(sequences[I1]), len(groundtruth[song_index][3]))))
    
    #structure = simple_structure(sequences[I1], alignments[I1], MIN_LEN2, MIN_DIST2)
    #print(str(sequences[7]))
    # plot_matrix(segments_to_matrix(alignments[7],
    #     (len(sequences[7]),len(sequences[7]))), 'z1m.png')

    simplify_graph(alignments[7], len(sequences[7]))
    
    # structs = get_simple_structures(SONGS[song_index], sequences, alignments)
    # lstructs = get_laplacian_structures(SONGS[song_index], sequences)
    # #print(segment_file(get_paths(SONGS[song_index])[0][0]).shape)
    # flstructs = get_orig_laplacian_structs(SONGS[song_index])
    # #print(type(structs), type(lstructs))
    # evals = evaluate_hierarchies(SONGS[song_index], groundtruth[song_index], structs)
    # levals = evaluate_hierarchies2(SONGS[song_index], groundtruth[song_index], lstructs)
    # print(np.mean(evals, axis=0), np.mean([s.shape[0] for s in structs]))
    # print(np.mean(levals, axis=0), np.mean([s.shape[0] for s in lstructs]))
    # flevals = evaluate_hierarchies3(SONGS[song_index], groundtruth[song_index], flstructs)
    # print(np.mean(flevals, axis=0), np.mean([s.shape[0] for s in flstructs]))
    # #print(structure[:,:30])
    
    #print(sequences[I1])
    #print(structure[-1,:])
    #plot_matrix(smith_waterman(groundtruth[0][-1,:], sequences[I1])[1], 'sw1.png')
    #plot_matrix(smith_waterman(groundtruth[0][-1,:], structure[-1,:])[1], 'sw2.png')
    
    
    
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
