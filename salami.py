import os, mir_eval, subprocess
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime
from matplotlib import pyplot as plt
from corpus_analysis.util import multiprocess, plot_matrix, buffered_run,\
    plot_sequences, save_json, load_json
from corpus_analysis.features import extract_chords, extract_bars,\
    get_summarized_chords, get_summarized_chroma, load_beats, get_duration
from corpus_analysis.alignment.affinity import get_alignment_segments,\
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix,\
    matrix_to_segments
from corpus_analysis.structure.structure import simple_structure
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity,\
    to_levels, get_laplacian_struct_from_audio, get_laplacian_struct_from_affinity2,\
    get_smooth_affinity_matrix
from corpus_analysis.structure.eval import evaluate_hierarchy, simplify
from corpus_analysis.structure.hcomparison import lmeasure

corpus = '/Users/flo/Projects/Code/FAST/grateful-dead/structure/SALAMI/'
audio = corpus+'lma-audio/'
annotations = corpus+'salami-data-public/annotations/'
features = corpus+'features/'
output = 'salami/'
RESULTS = output+'results3.csv'
graphditty = '/Users/flo/Projects/Code/Kyoto/GraphDitty/SongStructure.py'

K_FACTOR = 10
MIN_LEN = 16
MIN_DIST = 1 # >= 1
MAX_GAPS = 6
MAX_GAP_RATIO = .4
MIN_LEN2 = 8
MIN_DIST2 = 1
PLOT_FRAMES = 2000

def get_available_songs():
    return np.unique([int(s.split('.')[0]) for s in os.listdir(audio)
        if os.path.splitext(s)[1] == '.mp3'])

def extract_features(audio):
    extract_chords(audio, features)
    extract_bars(audio, features)

def extract_all_features():
    audio_files = [os.path.join(audio, a) for a in os.listdir(audio)]
    multiprocess('extracting features', extract_features, audio_files, True)

def calculate_fused_matrix(audio):
    filename = audio.split('/')[-1].replace('.mp3', '')
    if not os.path.isfile(features+filename+'.mat'):
        subprocess.call(['python', graphditty, '--win_fac', str(-1),
            '--filename', audio, '--matfilename', features+filename+'.mat',
            '--jsonfilename', features+filename+'.json'])#,
            #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def calculate_fused_matrices():
    audio_files = [os.path.join(audio, a) for a in os.listdir(audio)
        if os.path.splitext(a)[1] == '.mp3']
    [calculate_fused_matrix(a) for a in audio_files]

def load_beatwise_chords(index):
    return get_summarized_chords(features+str(index)+'_bars.txt',
        features+str(index)+'_chords.json')

def get_audio(index):
    return os.path.join(audio, str(index)+'.mp3')

def get_beatwise_chroma(index):
    return get_summarized_chroma(get_audio(index),
        features+str(index)+'_bars.txt')

def get_beats(index):
    return load_beats(features+str(index)+'_bars.txt')

def load_salami(filename):
    "load SALAMI event format as labeled intervals"
    events, labels = mir_eval.io.load_labeled_events(filename)
    #parsed files often have multiple labels at 0 or end, which boundaries_to_intervals can't handle
    while events[0] == events[1]:
        events, labels = events[1:], labels[1:]
    while events[-2] == events[-1]:
        events, labels = events[:-1], labels[:-1]
    #print(events, labels)
    intervals = mir_eval.util.boundaries_to_intervals(events)
    return intervals, labels[:len(intervals)]

def load_salami_hierarchy(index, annotation):
    prefix = annotations+str(index)+'/parsed/textfile'+str(annotation)+'_'
    files = [prefix+'uppercase.txt', prefix+'lowercase.txt']
    if all([os.path.isfile(f) for f in files]):
        intervals, labels = zip(*[load_salami(f) for f in files])
        return intervals, labels

def homogenize_labels(salami_hierarchy):
    labels = [[l.replace("'", '') for l in lev] for lev in salami_hierarchy[1]]
    uniq_labels = np.unique([l for l in np.concatenate(labels)])
    return (salami_hierarchy[0],
        [[np.where(uniq_labels == l)[0][0] for l in lev] for lev in labels])

def beatwise(salami_hierarchy, beats):
    salami_hierarchy = [s for s in salami_hierarchy[0] if len(s) > 0],\
        [[int(i) for i in s] for s in salami_hierarchy[1] if len(s) > 0]
    beat_intervals = list(zip(beats[:-1], beats[1:]))
    values = []
    for intervals, labels in zip(*salami_hierarchy):
        values.append([])
        for b in beat_intervals:
            indices = np.where(b[0] >= intervals[:,0])[0]
            values[-1].append(labels[indices[-1]] if len(indices) > 0 else -1)
    return np.array(values)
    # return np.array([[labels[np.where(b[0] >= intervals[:,0])[0][-1]]
    #     for b in beat_intervals])

def load_salami_hierarchies(index):
    hierarchies = [load_salami_hierarchy(index, a) for a in [1,2]]
    return [h for h in hierarchies if h != None]

def load_fused_matrix(index):
    m = sio.loadmat(features+str(index)+'.mat')
    j = load_json(features+str(index)+'.json')
    m = np.array(m['Ws']['Fused'][0][0])
    m[m < 0.01] = 0
    m[m != 0] = 1
    beats = j['times'][:len(m)]
    #plot_matrix(m)
    return m, beats

def test_eval():
    ref_hier, ref_lab = load_salami_hierarchy(10, 1)
    est_hier, est_lab = load_salami_hierarchy(10, 2)
    scores = mir_eval.hierarchy.evaluate(ref_hier, ref_lab, est_hier, est_lab)
    print(dict(scores))

def plot_hierarchy(path, index, method_name, intervals, labels, groundtruth):
    filename = path+str(index)+method_name+'.png'
    if not os.path.isfile(filename):
        maxtime = np.max(np.concatenate(groundtruth[0][0]))
        frames = np.linspace(0, int(maxtime), PLOT_FRAMES, endpoint=False)
        labelseqs = beatwise((intervals, labels), frames)
        if len(labelseqs) > 0:
            plot_sequences(labelseqs, path+str(index)+method_name+'.png')

def result_exists(groundtruth, method_name, index):
    exists = lambda data, col: (data[data.columns[:len(col)]] == col).all(1).any()
    if os.path.isfile(RESULTS):
        data = pd.read_csv(RESULTS)
        return all([exists(data, [index, K_FACTOR, MIN_LEN, MIN_DIST,
            MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2, i, method_name])
            for i in range(len(groundtruth))])

def eval_and_add_results(index, method_name, groundtruth, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index,
            K_FACTOR, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2,
            i, method_name, score[0], score[1], score[2]])
    print(results)
    results = pd.DataFrame(np.array(results),
        columns=['SONG', 'K_FACTOR', 'MIN_LEN', 'MIN_DIST', 'MAX_GAPS',
            'MAX_GAP_RATIO', 'MIN_LEN2', 'MIN_DIST2',
            'REF', 'METHOD', 'P', 'R', 'L'])
    if not os.path.isfile(RESULTS):
        results.to_csv(RESULTS, index=False)
    else:
        data = pd.read_csv(RESULTS)
        data = data.append(results, ignore_index=True)
        data.to_csv(RESULTS, index=False)

def eval_laplacian(index, groundtruth, method_name, plot_path=None, matrix=None, beats=None):
    if not result_exists(groundtruth, method_name, index):
        if matrix is not None and beats is not None:
            lpi, lpl = get_laplacian_struct_from_affinity2(matrix, beats)
        else:
            lpi, lpl = get_laplacian_struct_from_audio(get_audio(index))
        if plot_path:
            plot_hierarchy(plot_path, index, method_name, lpi, lpl, groundtruth)
        eval_and_add_results(index, method_name, groundtruth, lpi, lpl)

def eval_transitive(index, groundtruth, matrix, beats, method_name, plot_path=None):
    if not result_exists(groundtruth, method_name, index):
        alignment = matrix_to_segments(matrix)
        #plot_matrix(segments_to_matrix(alignment, (len(chroma),len(chroma))), 'est2.png')
        hierarchy = simple_structure(matrix[0], alignment, MIN_LEN2, MIN_DIST2)
        print(index, len(hierarchy), [len(h) for h in hierarchy])
        maxtime = np.max(np.concatenate(groundtruth[0][0]))
        beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
        hi, hl = [beat_ints for h in range(len(hierarchy))], hierarchy.tolist()
        if plot_path:
            plot_hierarchy(plot_path, index, method_name, hi, hl, groundtruth)
        eval_and_add_results(index, method_name, groundtruth, hi, hl)

def eval_hierarchy(index, matrix_method='fused', use_mat_in_lapl=True,
        plot_path='salami/all/', hom_labels=False):
    tname = 't_'+matrix_method
    lname = 'l'+(('_2'+matrix_method) if use_mat_in_lapl else '')
    groundtruth = load_salami_hierarchies(index)
    if hom_labels:
        groundtruth = [homogenize_labels(g) for g in groundtruth]
    if plot_path:
        if not hom_labels:#:D
            groundtruth = [homogenize_labels(g) for g in groundtruth]
        for i,g in enumerate(groundtruth):
            plot_hierarchy(plot_path, index, 'a'+str(i+1), g[0], g[1], groundtruth)
    if len(groundtruth) > 0 and not (result_exists(groundtruth, tname, index)
            and result_exists(groundtruth, lname, index)):
        if matrix_method == 'fused':
            matrix, beats = load_fused_matrix(index)
        elif matrix_method == 'mcfee':
            matrix, beats = get_smooth_affinity_matrix(get_audio(index))
        else:
            chroma = get_beatwise_chroma(index)#load_beatwise_chords(index)
            matrix, raw = get_affinity_matrix(chroma, chroma, False, MAX_GAPS,
                MAX_GAP_RATIO)#, k_factor=K_FACTOR)
            # plot_matrix(raw, 'est0.png')
            # plot_matrix(matrix, 'est1.png')
            beats = get_beats(index)
        
        alignment = get_segments_from_matrix(matrix, True, 0, MIN_LEN,
            MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)
        print(index, len(alignment))
        beats = beats[:len(matrix)]
        
        if not use_mat_in_lapl:
            eval_laplacian(index, groundtruth, lname, plot_path)
        else:
            eval_laplacian(index, groundtruth, lname, plot_path, matrix, beats)
        
        matrix = segments_to_matrix(alignment, (len(matrix), len(matrix)))
        # plot_matrix(matrix, 'est2.png')
        eval_transitive(index, groundtruth, matrix, beats, tname, plot_path)

def sweep():
    #print(get_available_songs()[197:222])
    multiprocess('evaluating hierarchies', eval_hierarchy,
        get_available_songs()[197:222], False)#[197:347])#[6:16])
    #print([mean([ r[0]] for r in result])

def plot():
    data = pd.read_csv(RESULTS)
    data = data[1183 <= data['SONG']][data['SONG'] <= 1211]
    #data = data[data['MIN_LEN'] == 24]
    #data = data[(data['K_FACTOR'] == 5) | (data['K_FACTOR'] == 10)]
    #data.groupby(['METHOD']).mean().T.plot(legend=True)
    data.groupby(['METHOD']).boxplot(column=['P','R','L'])
    #data.boxplot(column=['P','R','L'], by=['METHOD'])
    plt.show()

def test_eval(index):
    gti, gtl = load_salami_hierarchies(index)[0]
    hgti, hgtl = homogenize_labels((gti, gtl))
    lpi, lpl = get_laplacian_struct_from_audio(get_audio(index))
    print(evaluate_hierarchy(gti, gtl, lpi, lpl))
    print(evaluate_hierarchy(hgti, hgtl, lpi, lpl))

def get_intervals(levels, grid=None):
    basis = grid if grid is not None else np.arange(len(levels[0]))
    level = np.stack([basis[:-1], basis[1:]]).T
    return np.array([level for l in levels])

def test_eval_detail(index):
    refint, reflab = load_salami_hierarchies(index)[0]
    chroma = get_beatwise_chroma(index)
    affinity = get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO)[0]
    lapstruct = get_laplacian_struct_from_affinity(affinity)
    beats = get_beats(index)
    ints = get_intervals(lapstruct, beats)
    ints2, lapstruct2 = get_laplacian_struct_from_affinity2(affinity, beats)
    print(datetime.now().strftime("%H:%M:%S"))
    print(evaluate_hierarchy(refint, reflab, ints, lapstruct))
    print(datetime.now().strftime("%H:%M:%S"))
    print(evaluate_hierarchy(refint, reflab, ints, lapstruct2))
    print(datetime.now().strftime("%H:%M:%S"))

# for s in get_available_songs():
#     load_salami_hierarchies(s)
#test_hierarchy(get_available_songs()[0])
#run()
#print(get_available_songs()[297:])
#sweep()
#INDEX = 955
#eval_hierarchy(1208)#982)
#load_fused_matrix(1319)
#calculate_fused_matrices()
#test_hierarchy(INDEX)
plot()
#print(beatwise(homogenize_labels(load_salami_hierarchies(957)[0]), get_beats(957)))
#print(load_salami_hierarchies(972))
#load_salami_hierarchy(1003, 1)
#extract_all_features()