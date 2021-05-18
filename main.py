import os, json, random, itertools
import numpy as np
import pandas as pd
import graph_tool
from corpus_analysis.features import get_summarized_chords, to_multinomial, extract_essentia,\
    load_leadsheets, get_summarized_chords2, get_beat_summary, get_summarized_chroma,\
    get_summarized_mfcc, extract_chords, load_essentia
from corpus_analysis.stats.double_time import check_double_time
from corpus_analysis.stats.outliers import remove_outliers
from corpus_analysis.alignment.affinity import get_alignment_segments, get_affinity_matrix,\
    get_alignment_matrix, segments_to_matrix
from corpus_analysis.alignment.multi_alignment import align_sequences
from corpus_analysis.alignment.smith_waterman import smith_waterman
from corpus_analysis.util import profile, plot_matrix, plot_hist, plot,\
    buffered_run, multiprocess, plot_graph, boxplot, plot_sequences, flatten,\
    load_json, save_json
from corpus_analysis.structure.hcomparison import get_relative_meet_triples, get_meet_matrix
from corpus_analysis.structure.structure import shared_structure, simple_structure,\
    new_shared_structure, get_old_eval, get_new_eval, get_msa_eval
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity,\
    get_laplacian_struct_from_audio
from corpus_analysis.structure.graphs import graph_from_matrix, adjacency_matrix, alignment_graph
from corpus_analysis.structure.pattern_graph import PatternGraph, super_alignment_graph
from corpus_analysis.structure.eval import evaluate_hierarchy_varlen
from corpus_analysis.structure.hierarchies import get_hierarchy_labels, get_most_salient_labels
from gd import get_versions_by_date, get_paths, SONGS, get_chord_sequences

DATA = 'data/'
RESULTS = 'resultsNN/'
SONG_INDEX = 3
#alignment
SEG_COUNT = 0 #0 for all segments
MIN_LEN = 16
MIN_DIST = 1 # >= 1
MAX_GAPS = 4
MAX_GAP_RATIO = .2
NUM_MUTUAL = 20
MIN_LEN2 = 5
MIN_DIST2 = 4
L_MAX_GAPS = 4
L_MAX_GAP_RATIO = .2



def get_self_alignment(sequence):
    return get_alignment_segments(sequence, sequence, SEG_COUNT,
        MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)

def get_self_alignments(sequences):
    return multiprocess('self-alignments', get_self_alignment, sequences)

def get_pairings(sequences, num_mutual=NUM_MUTUAL):
    matrix = np.ones((len(sequences),len(sequences)), dtype=int)
    np.fill_diagonal(matrix, 0)
    for i in range(len(matrix)-1):
        places = np.where(np.sum(matrix, axis=0) > num_mutual)[0]
        places = places[places > i]
        num = min(np.sum(matrix[i])-num_mutual, len(places))
        if num > 0:
            choice = np.random.choice(places, num, replace=False)
            matrix[i,choice] = matrix[choice,i] = 0
    return np.vstack(np.nonzero(np.triu(matrix))).T.tolist()

def get_mutual_alignment(pairing_n_sequences):
    pairing, sequences = pairing_n_sequences
    return get_alignment_segments(sequences[pairing[0]], sequences[pairing[1]],
        SEG_COUNT, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)

def get_mutual_alignments(sequences, pairings):
    return multiprocess('mutual alignments', get_mutual_alignment,
        [(p, sequences) for p in pairings])

def preprocess_sequences(sequences):
    previous = [np.hstack(sequences)]
    sequences = remove_outliers(check_double_time(sequences))
    while not any([np.array_equal(s, np.hstack(sequences)) for s in previous]):
        #plot_sequences(sequences, str(len(previous))+'.png')
        previous.append(np.hstack(sequences))
        sequences = remove_outliers(check_double_time(sequences, 50), 5)
    return sequences

def get_preprocessed_seqs(song):
    sequences = buffered_run(DATA+song+'-chords',
        lambda: get_chord_sequences(song))
    sequences = buffered_run(DATA+song+'-pp',
        lambda: preprocess_sequences(sequences))
    return sequences

def get_alignments(name, sequences, preprocess=False):
    if preprocess:
        sequences = buffered_run(DATA+name+'-pp',
            lambda: preprocess_sequences(sequences))
    extension = 'pp' if preprocess else ''
    selfs = buffered_run(DATA+name+'-salign'+extension,
        lambda: get_self_alignments(sequences),
        [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST])
    if NUM_MUTUAL > 0:
        pairings = buffered_run(DATA+name+'-pairs'+extension,
            lambda: get_pairings(sequences), [NUM_MUTUAL])
        mutuals = buffered_run(DATA+name+'-malign'+extension,
            lambda: get_mutual_alignments(sequences, pairings),
            [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL])
    msa = buffered_run(DATA+name+'-msa'+extension,
        lambda: align_sequences(sequences)[0])
    selfp = np.stack((np.arange(len(sequences)), np.arange(len(sequences)))).T
    pairings = np.concatenate((selfp, pairings)) if NUM_MUTUAL > 0 else selfp
    alignments = np.concatenate((selfs, mutuals)) if NUM_MUTUAL > 0 else selfs
    return sequences, pairings, alignments, msa

def get_song_alignments(song, preprocess=False):
    sequences = buffered_run(DATA+song+'-chords',
        lambda: get_chord_sequences(song))
    return get_alignments(song, sequences, preprocess)

def get_simple_structure(sequence_n_alignment):
    sequence, alignment, index = sequence_n_alignment
    print(index, 'started')
    s = simple_structure(sequence, alignment, MIN_LEN2, MIN_DIST2)
    print(index, 'done')
    return s

def get_simple_structures(song, sequences, alignments):
    return buffered_run(DATA+song+'-structs',
        lambda: multiprocess('simple structures', get_simple_structure,
        list(zip(sequences, alignments, range(len(alignments))))),
        [SEG_COUNT, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL,
            MIN_LEN2, MIN_DIST2])

def get_laplacian_structure(sequence):
    affinity = get_affinity_matrix(sequence, sequence, True,
        L_MAX_GAPS, L_MAX_GAP_RATIO)[0]
    return get_laplacian_struct_from_affinity(affinity)

def get_laplacian_structures(song, sequences):
    return buffered_run(DATA+song+'-lstructs',
        lambda: multiprocess('laplacian structures', get_laplacian_structure,
        sequences), [L_MAX_GAPS, L_MAX_GAP_RATIO])

def get_orig_laplacian_struct(song_n_audio_n_version):
    song, audio, version = song_n_audio_n_version
    chords = get_summarized_chords2(beat_times,
        get_feature_path(song, version)+'_gochords.json')
    return get_laplacian_struct_from_audio(audio, chords)

def get_orig_laplacian_structs(song):
    audio_files = get_paths(song)[0]
    versions = get_versions_by_date(song)[0]
    return buffered_run(DATA+song+'-olstructs',
        lambda: multiprocess('original laplacian structures', get_orig_laplacian_struct,
        list(zip(itertools.repeat(song), audio_files, versions))), [])

def evaluate_hierarchies(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy_varlen,
        [(reference, e) for e in estimates]), [SEG_COUNT, MAX_GAPS,
            MAX_GAP_RATIO, MIN_LEN, MIN_DIST, NUM_MUTUAL, MIN_LEN2, MIN_DIST2])

def evaluate_hierarchies2(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy_varlen,
        [(reference, e) for e in estimates]), [L_MAX_GAPS, L_MAX_GAP_RATIO])

def evaluate_hierarchies3(song, reference, estimates):
    return buffered_run(DATA+song+'-eval',
        lambda: multiprocess('evaluate hierarchies', evaluate_hierarchy_varlen,
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
    g = graph_from_matrix(segments_to_matrix(alignment, (length, length)))[0]
    print(g)
    #g = graph_tool.topology.transitive_closure(g)
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
    for c in cliques:
        incli.a = np.isin(g.get_vertices(), c)
        #print(c, np.isin(g.get_vertices(), c).shape, np.sum(np.isin(g.get_vertices(), c)))
        subgraph = graph_tool.all.GraphView(g, vfilt=incli)
        #print(subgraph)
        union = subgraph if not union else graph_tool.generation.graph_union(union, subgraph)
    
    graph_tool.all.graph_draw(union, #vertex_fill_color=cli,
        output_size=(1000, 1000), bg_color=[1,1,1,1], output='z1c2.png')
    plot_matrix(adjacency_matrix(union), 'z1u.png')

def test_chroma_based_structure():
    audio = get_paths(SONGS[SONG_INDEX])[0][I1]
    version = list(DATASET[SONGS[SONG_INDEX]].keys())[I1]
    beatsFile = get_feature_path(SONGS[SONG_INDEX], version)+'_madbars.json'
    chroma = get_summarized_chroma(audio, beatsFile)
    #mfcc = get_summarized_mfcc(audio, bars)
    #combined = np.hstack([chroma, mfcc])
    matrix = get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO)[0]
    #plot_matrix(matrix)
    alignment = get_alignment_segments(chroma, chroma, SEG_COUNT,
        10, MIN_DIST, 10, 1)
    #print(alignment)
    plot_matrix(segments_to_matrix(alignment))
    structure = simple_structure(chroma, alignment, MIN_LEN2, MIN_DIST2)


def save_msa_eval(song, outfile):
    method = 'old2'
    data = pd.read_csv(outfile) if os.path.isfile(outfile) else pd.DataFrame([],
        columns=['song','method','entropy','partition count','total points'])
    if not ((data['song'] == song) & (data['method'] == method)).any():
        sequences, pairings, alignments, msa = get_song_alignments(song, False)
        # eval = get_msa_eval(RESULTS+song, sequences, msa)
        eval = get_old_eval(RESULTS+song, sequences, pairings, alignments,
            msa, MIN_LEN, MIN_DIST)
        #eval = get_new_eval(RESULTS+song, sequences, pairings, alignments)
        data = pd.read_csv(outfile)
        data.loc[len(data)] = [song, method]+list(eval)
        data.to_csv(outfile, index=False)

def multisong_set():
    sequences = flatten([get_chord_sequences(s)[:3] for s in SONGS[:5]], 1)
    #NUM_MUTUAL = 15
    sequences, pairings, alignments, msa = get_alignments('mixed', sequences, False)
    new_shared_structure(RESULTS+'mixed', sequences, pairings, alignments)

def run():
    #plot_date_histogram()
    # sequences, pairings, alignments, msa = get_song_alignments(SONGS[2])
    # plot_sequences(sequences, '0.png')
    # previous = [np.hstack(sequences)]
    # sequences = remove_outliers(check_double_time(sequences))
    # while not any([np.array_equal(s, np.hstack(sequences)) for s in previous]):
    #     plot_sequences(sequences, str(len(previous))+'.png')
    #     previous.append(np.hstack(sequences))
    #     sequences = remove_outliers(check_double_time(sequences, 50), 5)
    
    # for s in SONGS:
    #     # sequences, pairings, alignments, msa = get_song_alignments(s, True)
    #     # new_shared_structure(RESULTS+s, sequences, pairings, alignments)
    #     save_msa_eval(s, 'eval2.csv')
    #save_msa_eval(SONGS[SONG_INDEX], 'evallll.csv')
    
    multisong_set()
    
    # sequences, pairings, alignments, msa = get_song_alignments(SONGS[SONG_INDEX], True)
    # new_shared_structure(RESULTS+SONGS[SONG_INDEX], sequences, pairings, alignments)
    
    #save_msa_eval(SONGS[1], 'eval.json')
    
    #shared_structure(sequences, pairings, alignments, msa, MIN_LEN, MIN_DIST)
    
    # plot_matrix(get_affinity_matrix(sequences[0], sequences[0], True, MAX_GAPS, MAX_GAP_RATIO)[0], 'aff2.png')
    # plot_matrix(get_affinity_matrix(sequences[0], sequences[0], True, MAX_GAPS, MAX_GAP_RATIO)[1], 'aff0.png')
    
    #plot_msa(SONGS[SONG_INDEX], sequences, msa)
    #plot_evolution(SONGS[SONG_INDEX])
    
    #print(len(np.concatenate(alignments[7])), len(np.concatenate(alignments[8])))
    #print(sorted([len(a) for a in alignments]))
    
    #PatternGraph(sequences, pairings, alignments)
    #super_alignment_graph(SONGS[SONG_INDEX], sequences, pairings, alignments)
    #profile(lambda: new_shared_structure(RESULTS+SONGS[SONG_INDEX], sequences, pairings, alignments))
    # print(get_hierarchy_labels(sequences[:30])[0])
    # profile(lambda: get_most_salient_labels(sequences, 1, [9]))
    
    #align twice
    # plot_sequences(sequences.copy(), 'seqpat*.png')
    # sequences = super_alignment_graph(sequences, pairings, alignments)
    # plot_sequences(sequences.copy(), 'seqpat**.png')
    # selfs = multiprocess('self-alignments', get_self_alignment, sequences)
    # pairings = get_pairings(sequences)
    # mutuals = multiprocess('mutual alignments', get_mutual_alignment,
    #     [(p, sequences) for p in pairings])
    # selfp = np.stack((np.arange(len(sequences)), np.arange(len(sequences)))).T
    # pairings = np.concatenate((selfp, pairings))
    # alignments = np.concatenate((selfs, mutuals))
    # sequences = super_alignment_graph(sequences, pairings, alignments)
    # plot_sequences(sequences.copy(), 'seqpat***.png')
    
    #plot_matrix(segments_to_matrix(alignments[7]))
    # plot_matrix(segments_to_matrix(
    #     get_alignment_segments(sequences[7], sequences[7], SEG_COUNT, MIN_LEN, MIN_DIST, 4, .2),
    #     (len(sequences[7]), len(sequences[7]))))
    #shared_structure(sequences, pairings, alignments, msa, MIN_LEN, MIN_DIST)
    #profile(lambda: shared_structure(sequences, sas, multinomial, msa))
    groundtruth = load_leadsheets(leadsheets, SONGS)
    #print("ground", groundtruth[SONG_INDEX])
    #plot_matrix(get_alignment_matrix(sequences[62], sequences[62],
    #    MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO), 's4.png')
    #print("done")
    
    I1, I2 = 62, 62#7, 62
    
    #print(sequences[I1][:100], groundtruth[SONG_INDEX][3][:100])
    # plot_matrix(segments_to_matrix(
    #     [smith_waterman(sequences[I1], groundtruth[SONG_INDEX][3])],
    #     (len(sequences[I1]), len(groundtruth[SONG_INDEX][3]))))
    
    
    
    
    
    
    #plot_matrix(segments_to_matrix(alignments[I1],
    #    (len(sequences[I1]),len(sequences[I1]))), 'est1.png')
    
    #####structure = simple_structure(sequences[I1], alignments[I1], MIN_LEN2, MIN_DIST2)
    
    #print(str(sequences[I1]))

    #simplify_graph(alignments[7], len(sequences[7]))
    
    #structs = get_simple_structures(SONGS[SONG_INDEX], sequences, alignments)
    # lstructs = get_laplacian_structures(SONGS[SONG_INDEX], sequences)
    # #print(segment_file(get_paths(SONGS[SONG_INDEX])[0][0]).shape)
    # flstructs = get_orig_laplacian_structs(SONGS[SONG_INDEX])
    # #print(type(structs), type(lstructs))
    # evals = evaluate_hierarchies(SONGS[SONG_INDEX], groundtruth[SONG_INDEX], structs)
    # levals = evaluate_hierarchies2(SONGS[SONG_INDEX], groundtruth[SONG_INDEX], lstructs)
    # depths = lambda st: np.array([s.shape[0] for s in st])
    # print(np.mean(evals, axis=0), np.mean(depths(structs)))
    # print(np.mean(levals, axis=0), np.mean(depths(lstructs)))
    # flevals = evaluate_hierarchies3(SONGS[SONG_INDEX], groundtruth[SONG_INDEX], flstructs)
    # print(np.mean(flevals, axis=0), np.mean(depths(flstructs)))
    # 
    # boxplot(np.hstack([levals, evals, flevals]), 'lmeasures.png')
    # boxplot(np.array([depths(lstructs), depths(structs), depths(flstructs)]).T, 'depths.png')
    #print(structure[:,:30])
    
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

#print(get_pairings([0,1,2,3,4,5,6,7], 5))
