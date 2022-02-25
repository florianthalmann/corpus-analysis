import os, mir_eval, subprocess, tqdm, gc, optuna, librosa, itertools
from multiprocessing import Pool, cpu_count
from mutagen.mp3 import MP3
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import block_diag
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from matplotlib import pyplot as plt
from corpus_analysis.util import multiprocess, plot_matrix, buffered_run,\
    plot_sequences, save_json, load_json, flatten, catch, RepeatPruner, profile,\
    summarize_matrix, plot_hist
from corpus_analysis.features import extract_chords, extract_bars,\
    get_summarized_chords, get_summarized_chroma, load_beats, get_summarized_mfcc
from corpus_analysis.alignment.affinity import get_alignment_segments,\
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix,\
    matrix_to_segments, threshold_matrix, get_best_segments, ssm,\
    double_smooth_matrix, peak_threshold
from corpus_analysis.structure.structure import simple_structure
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity2,\
    to_levels, get_laplacian_struct_from_audio, get_smooth_affinity_matrix
from corpus_analysis.structure.graphs import blockmodel
from corpus_analysis.structure.eval import evaluate_hierarchy, simplify
from corpus_analysis.stats.hierarchies import monotonicity, label_monotonicity,\
    interval_monotonicity, beatwise_ints, strict_transitivity, order_transitivity,\
    relabel_adjacent, to_int_labels, auto_labeled, repetitiveness, complexity
from corpus_analysis.stats.util import entropy
from corpus_analysis.stats.matrices import fractal_dimension
from corpus_analysis.structure.hierarchies import divide_hierarchy,\
    make_segments_hierarchical, add_transitivity_to_matrix
from corpus_analysis.structure.novelty import get_novelty_boundaries
from corpus_analysis.data import Data
import experiments.matrix_analysis as matrix_analysis

# PARAMS = dict([
#     ['MATRIX_TYPE', 2],#0=own, 1=mcfee, 2=fused, 3=own2
#     ['MEDIAN_LEN', 16],
#     ['SIGMA', 1.004],
#     ['WEIGHT', 0.5],#how much mfcc: 1 = only
#     ['THRESHOLD', 0],
#     ['NUM_SEGS', 100],
#     ['MIN_LEN', 6],
#     ['MIN_DIST', 1],
#     ['MAX_GAPS', 8],
#     ['MAX_GAP_RATIO', .5],
#     ['MIN_LEN2', 10],
#     ['MIN_DIST2', 1],
#     ['LEXIS', 1],
#     ['BETA', 0.3]
# ])
PARAMS = dict([
    ['MATRIX_TYPE', 2],#0=own, 1=mcfee, 2=fused, 3=own2
    ['SEGMENT_TYPE', 0],#0=best, 1=old
    ['MEDIAN_LEN', 16],
    ['SIGMA', 0.016],
    ['WEIGHT', 0.5],#how much mfcc: 1 = only
    ['THRESHOLD', 0],
    ['NUM_SEGS', 100],
    ['MIN_LEN', 10],
    ['MIN_DIST', 1],
    ['MAX_GAPS', 0],
    ['MAX_GAP_RATIO', .2],
    ['MIN_LEN2', 10],
    ['MIN_DIST2', 1],
    ['LEXIS', 1],
    ['BETA', 0.5],
    ['ALT', 0]#segment count threshold below which to recalculate matrix
])
# PARAMS = dict([
#     ['MATRIX_TYPE', 0],#0=own, 1=mcfee, 2=fused, 3=own2
#     ['MEDIAN_LEN', 16],
#     ['SIGMA', 0.016],
#     ['WEIGHT', 0.5],#how much mfcc: 1 = only
#     ['THRESHOLD', 2],
#     ['NUM_SEGS', 100],
#     ['MIN_LEN', 10],
#     ['MIN_DIST', 1],
#     ['MAX_GAPS', 0],
#     ['MAX_GAP_RATIO', .2],
#     ['MIN_LEN2', 10],
#     ['MIN_DIST2', 1],
#     ['LEXIS', 1],
#     ['BETA', 0.2]
# ])
#{'t': 0, 'k': 3.0, 'n': 100, 'ml': 20, 'md': 1, 'mg': 5, 'mgr': 0.5, 'ml2': 20, 'md2': 1, 'lex': 1, 'beta': 0.2}

#1.489692 {'t': 2, 'k': 3.89, 'n': 100, 'ml': 10, 'md': 2, 'mg': 13, 'mgr': 0.775, 'ml2': 27, 'md2': 1, 'lex': 1, 'beta': 0.343}
# PARAMS = dict([
#     ['MATRIX_TYPE', 2],#0=own, 1=mcfee, 2=fused, 3=own2
#     ['WEIGHT', 0.5],
#     ['THRESHOLD', 1],
#     ['NUM_SEGS', 100],
#     ['MIN_LEN', 6],
#     ['MIN_DIST', 1],
#     ['MAX_GAPS', 13],
#     ['MAX_GAP_RATIO', .75],
#     ['MIN_LEN2', 25],
#     ['MIN_DIST2', 1],
#     ['LEXIS', 1],
#     ['BETA', 0.21]
# ])
#'k': 2.63, 'n': 100, 'ml': 6, 'md': 1, 'mg': 10, 'mgr': 0.736, 'ml2': 27, 'md2': 1, 'lex': 1, 'beta': 0.253

def matrix_type(params):
    index = params['MATRIX_TYPE']
    return 'own' if index == 0 else 'mcfee' if index == 1 else 'fused'

corpus = '/Users/flo/Projects/Code/Kyoto/SALAMI/'
audio = corpus+'all-audio'#'lma-audio/'
annotations = corpus+'salami-data-public/annotations/'
features = corpus+'features/'
output = 'salami/'
DATA = output+'data/'
#RESULTS = Data(None,
RESULTS = Data(output+'resultsF34.csv',
    columns=['SONG']+list(PARAMS.keys())+['REF', 'METHOD', 'P', 'R', 'L'])
# RESULTS = Data(output+'lapl.csv',
#     columns=['SONG', 'LEVELS', 'REF', 'P', 'R', 'L'])
PLOT_PATH=''#output+'all22/'#''#output+'all12/'
graphditty = '/Users/flo/Projects/Code/Kyoto/GraphDitty/SongStructure.py'

PLOT_FRAMES = 2000

HOM_LABELS=False
METHOD_NAME='t'

#some annotations are missing!
def get_annotation_ids():
    return np.unique([int(a) for a in os.listdir(annotations) if a != '.DS_Store'])

def get_audio_files():
    return [os.path.join(audio, a) for a in os.listdir(audio)
        if os.path.splitext(a)[1] == '.mp3']

#get all salami_ids for which there are annotation files and an audio file
#whose length is longer than 30 seconds and corresponds to the annotation
def get_available_songs():
    audio_ids = [int(a.split('/')[-1].split('.')[0]) for a in get_audio_files()]
    durs = [catch(lambda a: MP3(a).info.length, lambda e: 0, a)
        for a in get_audio_files()]
    #average duration of groudtruths
    salami_duration = lambda i:\
        np.mean([a[0][-1][-1][-1] for a in load_salami_hierarchies(i)])
    sdurs = {i:salami_duration(i) for i in get_annotation_ids()}
    audio_ids = [a for a,d in zip(audio_ids, durs)
        if d >= 30 and a in sdurs and abs(d-sdurs[a]) < 1]
    print('kept', len(audio_ids), 'of', len(durs), 'salami files')
    return np.unique(audio_ids) #sort and make sure they're unique..

def extract_features(audio):
    #extract_chords(audio, features)
    extract_bars(audio, features, True)

def extract_all_features():
    multiprocess('extracting features', extract_features, get_audio_files(), True)

def calculate_fused_matrix(audio, force=False):
    filename = audio.split('/')[-1].replace('.mp3', '')
    if force or not os.path.isfile(features+filename+'.mat')\
            or not os.path.isfile(features+filename+'.json'):
        subprocess.call(['python', graphditty, '--win_fac', str(-2),#beats
            '--filename', audio, '--matfilename', features+filename+'.mat',
            '--jsonfilename', features+filename+'.json',
            '--K', str(3), '--reg_neighbs', str(0.0), '--niters', str(10),
            '--neigs', str(10), '--wins_per_block', str(2)])#,
            #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def calculate_fused_matrices():
    multiprocess('fusing matrices', calculate_fused_matrix, get_audio_files(), True)
    #[calculate_fused_matrix(a) for a in tqdm.tqdm(get_audio_files())]

def load_beatwise_chords(index):
    return get_summarized_chords(features+str(index)+'_bars.txt',
        features+str(index)+'_chords.json')

def get_audio(index):
    return os.path.join(audio, str(index)+'.mp3')

def get_beatwise_chroma(index, beats=None):
    return get_summarized_chroma(get_audio(index),
        features+str(index)+'_bars.txt', beats)

def get_beatwise_mfcc(index, beats=None):
    return get_summarized_mfcc(get_audio(index),
        features+str(index)+'_bars.txt', beats)

def get_beats(index):
    return load_beats(features+str(index)+'_bars.txt')

def load_salami(filename):
    "load SALAMI event format as labeled intervals"
    events, labels = mir_eval.io.load_labeled_events(filename)
    #parsed files often have multiple labels at 0 or end, which boundaries_to_intervals can't handle
    events, indices = np.unique(events, return_index=True)
    labels = [l for i,l in enumerate(labels) if i in indices][:-1]#end label not needed
    intervals = mir_eval.util.boundaries_to_intervals(events)
    return intervals, labels

def load_salami_hierarchy(index, annotation):
    prefix = annotations+str(index)+'/parsed/textfile'+str(annotation)+'_'
    files = [prefix+'uppercase.txt', prefix+'lowercase.txt']
    if all([os.path.isfile(f) for f in files]):
        intervals, labels = zip(*[load_salami(f) for f in files])
        return intervals, labels

def load_salami_hierarchies(index):
    hierarchies = [load_salami_hierarchy(index, a) for a in [1,2]]
    return [h for h in hierarchies if h != None]

def homogenize_labels(salami_hierarchy):
    return (salami_hierarchy[0],
        [[l.replace("'", '') for l in lev] for lev in salami_hierarchy[1]])

def int_labels(salami_hierarchy):
    labels = salami_hierarchy[1]
    uniq_labels = np.unique([l for l in np.concatenate(labels)])
    return (salami_hierarchy[0],
        [[np.where(uniq_labels == l)[0][0] for l in lev] for lev in labels])

def load_fused_matrix(index, params, threshold=True, var_sigma=True):
    prev = params['SIGMA']
    # if var_sigma:
    #     params['SIGMA'] = best_sigma(index)
    
    m = sio.loadmat(features+str(index)+'.mat')
    j = load_json(features+str(index)+'.json')
    m = np.array(m['Ws']['Fused MFCC/Chroma'][0][0])#['Fused'][0][0])
    #beats = np.array(j['times'][:len(m)])
    beats = get_beats(index)
    factor = round(len(m)/len(beats))
    m = summarize_matrix(m, factor)
    # m[m >= 0.05] = 0
    # plot_hist(np.hstack(m[m > 0.01]), bincount=100, path=PLOT_PATH+str(index)+'-hist.png')
    
    #m = np.log2(m)
    if PLOT_PATH: plot_matrix(m, PLOT_PATH+str(index)+'-m-f.png')
    if threshold:
        if params['THRESHOLD'] == 0:
            m = peak_threshold(m, params['MEDIAN_LEN'], params['SIGMA'])
            m = np.logical_or(m.T, m)#thresholding may lead to asymmetries (peak picking..)
        else:
            m = threshold_matrix(m, params['THRESHOLD'])
        m = np.triu(m, k=1)#now symmetrix so only keep upper triangle
    # if len(beats)+1 != len(m):
    #     print("BEATS DONT MATCH!!!!", len(beats), len(m))
    params['SIGMA'] = prev#to maintain uniform data record for experiments
    return m, beats

def get_monotonic_salami():
    annos = {i:load_salami_hierarchies(i) for i in get_available_songs()}
    return [i for i in annos.keys() if all([interval_monotonicity(h) for h in annos[i]])]

def plot_hierarchy(path, index, method_name, intervals, labels, groundtruth, force=True):
    filename = path+str(index)+method_name+'.png'
    if force or not os.path.isfile(filename):
        maxtime = np.max(np.concatenate(groundtruth[0][0]))
        frames = np.linspace(0, int(maxtime), PLOT_FRAMES, endpoint=False)
        labelseqs = beatwise_ints((intervals, labels), frames)
        if len(labelseqs) > 0:
            plot_sequences(labelseqs, path+str(index)+method_name+'.png')

def plot_groundtruths(groundtruth, index, plot_path):
    groundtruth = [int_labels(v) for v in groundtruth]
    for j,v in enumerate(groundtruth):
        plot_hierarchy(plot_path, index, 'a'+str(j+1)+'h', v[0], v[1], groundtruth)

def evaluate(index, method_name, param_vals, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(get_groundtruth(index)):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index]+param_vals+[i, method_name, score[0], score[1], score[2]])
    print(results)
    return results

def own_chroma_affinity(index, params, beats=None):
    chroma = beatwise_feature(index, 'chroma', get_beatwise_chroma, beats)
    return own_affinity(index, params, [chroma], [1], beats)

def own_mfcc_affinity(index, params, beats=None):
    mfcc = beatwise_feature(index, 'mfcc', get_beatwise_mfcc, beats)
    return own_affinity(index, params, [mfcc], [1], beats)

def own_chroma_mfcc_affinity(index, params, beats=None):
    chroma = beatwise_feature(index, 'chroma', get_beatwise_chroma, beats)
    mfcc = beatwise_feature(index, 'mfcc', get_beatwise_mfcc, beats)
    return own_affinity(index, params, [chroma, mfcc],
        [1-params['WEIGHT'], params['WEIGHT']], beats)

def beatwise_feature(index, name, func, beats=None):
    return buffered_run(DATA+name+str(index), lambda: func(index),
        [len(beats)] if beats is not None else [])#load_beatwise_chords(index)

#features is an array of arrays of feature vectors (e.g. chroma, mfcc,...)
def own_affinity(index, params, features, weights, beats=None):
    features = [w * MinMaxScaler().fit_transform(f)
        for f,w in zip(features, weights)]
    mix = np.hstack(features) if len(features) > 1 else features[0]
    matrix, raw = get_affinity_matrix(mix, mix, False, params['MAX_GAPS'],
        params['MAX_GAP_RATIO'], params['THRESHOLD'])
    if beats is None: beats = get_beats(index)
    return matrix, raw, beats

def own_chroma_affinity_new(index):
    chroma = buffered_run(DATA+'chroma'+str(index),
        lambda: get_beatwise_chroma(index))#load_beatwise_chords(index)
    chroma = MinMaxScaler().fit_transform(chroma)
    matrix, beats = ssm(chroma, chroma), get_beats(index)
    raw = threshold_matrix(matrix, params['THRESHOLD'])
    return matrix, raw, beats

def labels_to_hierarchy(index, labels, target, beats, groundtruth, divide=False):
    maxtime = np.max(np.concatenate(groundtruth[0][0]))
    beats = beats[:target.shape[0]]#just to make sure
    beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
    hierarchy = [beat_ints for h in range(len(labels))], labels.tolist()
    if divide:#divide the hierarchy where appropriate based on mfccs
        ssm = own_mfcc_affinity(index, beats)[0]
        boundaries = get_novelty_boundaries(ssm)
        print(boundaries)
        hierarchy = divide_hierarchy(boundaries, hierarchy)
    return hierarchy

def get_groundtruth(index):
    groundtruth = load_salami_hierarchies(index)
    if HOM_LABELS: groundtruth = [homogenize_labels(v) for v in groundtruth]
    if PLOT_PATH: plot_groundtruths(groundtruth, index, PLOT_PATH)
    return groundtruth

def divide_matrix(index, beats, matrix, raw):
    ssm = own_mfcc_affinity(index, beats)[0]
    #ssm = load_fused_matrix(index, PARAMS, False)[0]
    boundaries = get_novelty_boundaries(ssm)-1#, kernel_size=PARAMS['MEDIAN_LEN'], sigma=PARAMS['SIGMA'])-1
    boundaries = boundaries[boundaries >= 0]
    print(boundaries)
    if len(boundaries) > 0:
        matrix[boundaries] = 0
        matrix.T[boundaries] = 0
        raw[boundaries] = 0
        raw.T[boundaries] = 0
    return matrix, raw

def get_matrices(index, params):
    if params['MATRIX_TYPE'] is 2:
        matrix, beats = load_fused_matrix(index, params)
        raw = matrix.copy()
    elif params['MATRIX_TYPE'] is 1:
        matrix, beats = get_smooth_affinity_matrix(get_audio(index))
        raw = matrix.copy()
    elif params['MATRIX_TYPE'] is 3:
        matrix, raw, beats = own_chroma_affinity_new(index, params)
    else:
        matrix, raw, beats = own_chroma_affinity(index, params)
    
    #matrix, raw = divide_matrix(index, beats, matrix, raw)
    
    params = params.copy()
    if 0 < params['MIN_LEN'] < 1:#relative minlen
        params['MIN_LEN'] = round(matrix.shape[0]*params['MIN_LEN'])
        print('relative minlen', matrix.shape, params['MIN_LEN'])
    
    if params['SEGMENT_TYPE'] == 0:
        smatrix = get_best_segments(matrix, params['MIN_LEN'],
            min_dist=params['MIN_DIST'], min_val=1-params['MAX_GAP_RATIO'],
            max_gap_len=params['MAX_GAPS'])
    else:
        smatrix = segments_to_matrix(get_segments_from_matrix(matrix, True,
            params['NUM_SEGS'], params['MIN_LEN'], params['MIN_DIST'],
            params['MAX_GAPS'], params['MAX_GAP_RATIO'], raw), matrix.shape)
    
    if PLOT_PATH:
        plot_matrix(raw, PLOT_PATH+str(index)+'-m0'+matrix_type(params)[0]+'.png')
        plot_matrix(matrix, PLOT_PATH+str(index)+'-m1'+matrix_type(params)[0]+'.png')
        plot_matrix(smatrix, PLOT_PATH+str(index)+'-m2'+matrix_type(params)[0]+'.png')
    
    return smatrix, matrix, raw, beats

def get_matrices_buf(index, params, path=DATA):
    return buffered_run(path+'matrices'+str(index),
        lambda: get_matrices(index, params), [params[p] for p in params
            if p not in ('MIN_LEN2','MIN_DIST2','BETA','LEXIS','ALT')])

def get_hierarchy(index, params):
    groundtruth = get_groundtruth(index)
    smatrix, matrix, raw, beats = get_matrices_buf(index, params)
    segments = matrix_to_segments(smatrix)
    # sorted(segments, key=lambda s: len(s), reverse=True)
    # segments = segments[:params['NUM_SEGS']]
    
    if len(segments) < params['ALT']:
        print('alternative matrix!')
        params = params.copy()
        if params['MATRIX_TYPE'] == 2:
            params['SIGMA'] = 0
        elif params['THRESHOLD'] < 10:
            params['THRESHOLD'] *= 2
        else:
            params['THRESHOLD'] -= 0.05
        smatrix, matrix, raw, beats = get_matrices_buf(index, params)
        segments = matrix_to_segments(smatrix)
    
    target = raw#smatrix
    
    if 0 < params['MIN_LEN2'] < 1:#relative minlen2
        params['MIN_LEN2'] = round(matrix.shape[0]*params['MIN_LEN2'])
        print('relative minlen2', matrix.shape, params['MIN_LEN2'])
    plot_file = PLOT_PATH+str(index)+'-m3'+matrix_type(params)[0]+'.png' if PLOT_PATH else None
    labels, hmatrix = simple_structure(segments, params['MIN_LEN2'],
        params['MIN_DIST2'], params['BETA'], target, lexis=params['LEXIS'] == 1,
        plot_file=plot_file)
    
    hierarchy = labels_to_hierarchy(index, labels, target, beats, groundtruth)
    
    if PLOT_PATH: plot_hierarchy(PLOT_PATH, index, 'o'+matrix_type(params)[0],
        hierarchy[0], hierarchy[1], groundtruth, force=True)
    return hierarchy, labels, hmatrix

def get_hierarchy_buf(index, params, path=DATA):
    return buffered_run(path+'hierarchy'+str(index),
        lambda: get_hierarchy(index, params), params.values())

#24 31 32 37 47 56   5,14   95  135 148 166     133     1627    231     618
def own_eval(args):
    index, method_name, params = args
    #calculate_fused_matrix(get_audio(index), True)
    t, l, m = get_hierarchy_buf(index, params)#, 'ownBUFFER')
    # print(t[0][:2])
    # print(t[1][:10])
    return evaluate(index, method_name, list(params.values()), t[0], t[1])

def lapl_eval(params):
    index, method_name = params[0], params[1]
    l = buffered_run(DATA+'lapl'+str(index),#+method_name+'-'+str(index),
        lambda: get_laplacian_struct_from_audio(get_audio(index)))
    if PLOT_PATH: plot_matrix(get_smooth_affinity_matrix(get_audio(index))[0], PLOT_PATH+str(index)+'-m0m.png')
    # num_levels = len(own[0])
    # l = l[0][:num_levels], l[1][:num_levels]
    if PLOT_PATH: plot_hierarchy(PLOT_PATH, index, method_name, l[0], l[1], get_groundtruth(index))
    # print(l[0][:2])
    # print(l[1][:10])
    return evaluate(index, method_name, [0 for v in list(params.values())], l[0], l[1])

def get_ref_rows(index, method_name, ignore_params):
    params = list(PARAMS.values())
    if ignore_params: params = [0 for v in params]
    return [[index]+params+[i, method_name]
        for i in range(len(load_salami_hierarchies(index)))]

def get_missing_indices(indices, method_name, ignore_params):
    ref_rows = [get_ref_rows(i, method_name, ignore_params) for i in indices]
    return [r[0][0] for r in ref_rows if not RESULTS.rows_exist(r)]

def multi_eval(indices, method_name, method_func, ignore_params, params=PARAMS):
    #calculate missing results
    missing = get_missing_indices(indices, method_name, ignore_params)
    l2params = [[i, method_name, params] for i in missing]
    with Pool(processes=cpu_count()-2) as pool:
        for r in tqdm.tqdm(pool.imap_unordered(method_func, l2params), total=len(l2params), desc=method_name):
            RESULTS.add_rows(r)
    #get all results
    return [RESULTS.get_rows(get_ref_rows(i, method_name, ignore_params)) for i in indices]

def comparative_eval(indices, params=PARAMS):
    lapl = [np.mean([e[-1] for e in s])
        for s in multi_eval(indices, 'l', lapl_eval, True, params)]
    own = [np.mean([e[-1] for e in s])
        for s in multi_eval(indices, 'tcN', own_eval, False, params)]
    print(lapl)
    print(own)
    print([o-l for l,o in zip(lapl, own)])
    return np.mean([o-l for l,o in zip(lapl, own)])

def eval_laplacian(index, levels, groundtruth, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index, levels, i, score[0], score[1], score[2]])
    print(results)
    return results

def run_laplacian(params):
    index, levels, results = params
    ref_rows = [[index, levels, i]
        for i in range(len(load_salami_hierarchies(index)))]
    if not results or not results.rows_exist(ref_rows):
        gt = load_salami_hierarchies(index)
        l = get_laplacian_struct_from_audio(get_audio(index), levels)
        if PLOT_PATH: plot_hierarchy(PLOT_PATH, index, 'l', l[0], l[1], gt)
        rows_func = lambda: eval_laplacian(index, levels, gt, l[0], l[1])
        print(print(eval_laplacian(index, levels, gt, l[0], l[1])))
        l = to_int_labels(l)
        l = l[0], np.array([relabel_adjacent(ll) for ll in l[1]])
        print(eval_laplacian(index, levels, gt, l[0], l[1]))
        if results:
            return results.add_rows(ref_rows, rows_func)

def sweep_laplacian(indices):
    for l in range(2, 40):
        params = [[i, l, RESULTS] for i in indices]
        multiprocess('multi eval', run_laplacian, params, True)

def plot_laplacian(path='lapl.png'):
    data = RESULTS.read()
    print(data.groupby(['LEVELS']).mean())
    data.boxplot(column=['P','R','L'], by=['LEVELS'])
    plt.tight_layout()
    plt.savefig(path, dpi=1000)

def plot(path=None):
    data = RESULTS.read()
    pd.set_option('display.max_rows',None)
    #data2 = data[np.isin(data['SONG'], [1099,1179,1210,1431,616,1419,578,749,765,1405,198,1603,1395,1059,696,774,1196,675,1186,1347,458,1648,244,1392,14])]
    print(data.groupby(['SONG','METHOD'])[['P','R','L']].mean())
    #print(nothing)
    #data = data[1183 <= data['SONG']][data['SONG'] <= 1211]
    #data = data[data['SONG'] <= 333]
    #data = data[data['MIN_LEN'] == 24]
    #data = data[(data['THRESHOLD'] == 5) | (data['THRESHOLD'] == 10)]
    #data.groupby(['METHOD']).mean().T.plot(legend=True)
    #data.groupby(['METHOD']).boxplot(column=['P','R','L'])
    print(data[data['METHOD'] == 'l'].groupby(['SONG','REF']).max().groupby(['SONG']).mean().mean())
    print(data[data['METHOD'] == 't'].groupby(['SONG','REF']).max().groupby(['SONG']).mean().mean())
    # print(data.groupby(['METHOD', 'MATRIX_TYPE']).mean())
    # print(data[data['METHOD'] == 'l'].groupby(['SONG']).max().groupby(['SONG']).mean())
    # print(data[data['METHOD'] == 't'].groupby(['SONG']).max().groupby(['SONG']).mean())
    #print(data[(data['METHOD'] == 't_ownNY') | (data['METHOD'] == 'l')].sort_values(['SONG', 'METHOD']).to_string())
    #print(data.groupby(['SONG', 'METHOD']).mean().sort_values(['SONG', 'METHOD']).to_string())
    #print(data[data['METHOD'] != 'l'].groupby(['SONG']).mean())
    data.boxplot(column=['P','R','L'], by=['METHOD'])
    #plt.show()
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()

def analyze(path='analysis'):
    data = RESULTS.read()
    data = data[data['METHOD'] == 't']
    songs = data['SONG'].unique()
    matrix_sizes = dict(zip(songs, [len(get_beats(s)) for s in songs]))
    data = data.sort_values('L', ascending=False).drop_duplicates(['SONG','REF'])
    data['BEATS'] = data['SONG'].map(matrix_sizes)
    print(data)
    plot(lambda: data.plot.scatter(x='BEATS', y='L'), path+'L.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='BETA'), path+'BETA.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='SIGMA'), path+'SIGMA.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='MIN_LEN'), path+'MIN_LEN.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='MAX_GAPS'), path+'MAX_GAPS.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='MAX_GAP_RATIO'), path+'MAX_GAP_RATIO.pdf')
    plot(lambda: data.plot.scatter(x='BEATS', y='MIN_LEN2'), path+'MIN_LEN2.pdf')
    
    plot(lambda: data.plot.scatter(x='MAX_GAP_RATIO', y='MAX_GAPS'), path+'_mgr_mg.pdf')
    plot(lambda: data.plot.scatter(x='BETA', y='SIGMA'), path+'_beta_sigma.pdf')
    plot(lambda: data.plot.scatter(x='BETA', y='L'), path+'_beta_l.pdf')
    plot(lambda: data.plot.scatter(x='SIGMA', y='L'), path+'_sigma_l.pdf')
    plot(lambda: data.plot.scatter(x='SIGMA', y='MAX_GAP_RATIO'), path+'_sigma_mgr.pdf')
    plot(lambda: data.plot.scatter(x='SIGMA', y='MAX_GAPS'), path+'_sigma_mg.pdf')
    
    data = RESULTS.read()
    # for s in songs:
    #     data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
    #     plot(lambda: data2.plot.scatter(x='SIGMA', y='L'), path+'SIGMA'+str(s)+'.pdf')
    # 
    # for s in songs:
    #     data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
    #     plot(lambda: data2.plot.scatter(x='MAX_GAPS', y='L'), path+'SMGR'+str(s)+'.pdf')
    for s in songs:
        data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
        plot(lambda: data2.plot.scatter(x='BETA', y='L'), path+'BETA'+str(s)+'.pdf')
    for s in songs:
        data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
        plot(lambda: data2.plot.scatter(x='MAX_GAP_RATIO', y='L'), path+'MGR'+str(s)+'.pdf')
    for s in songs:
        data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
        plot(lambda: data2.plot.scatter(x='MIN_LEN', y='L'), path+'ML'+str(s)+'.pdf')
    for s in songs:
        data2 = data[(data['METHOD'] == 't') & (data['SONG'] == s)]
        plot(lambda: data2.plot.scatter(x='MIN_LEN2', y='L'), path+'MLTWO'+str(s)+'.pdf')

#increasing number of possible pairs with same label decreases monotonicity...
#(denominator gets larger)
def label_count_test(hierarchy, beats):
    #double number of beats
    means = np.append(np.mean(np.vstack((beats[:-1], beats[1:])), axis=0), [0])
    beats2 = np.vstack((beats, means)).reshape((-1,), order='F')[:-1]
    print(label_monotonicity(hierarchy, beats),
        label_monotonicity(hierarchy, beats2))

def hierarchy_analysis(hiers, beats, csv_buffer):
    if csv_buffer and os.path.isfile(csv_buffer):
        data = pd.read_csv(csv_buffer)
    else:
        mi = [interval_monotonicity(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='mi')]
        ml = [label_monotonicity(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='ml')]
        to = [order_transitivity(h) for h in tqdm.tqdm(hiers, desc='to')]
        ts = [strict_transitivity(h) for h in tqdm.tqdm(hiers, desc='ts')]
        r = [repetitiveness(h) for h in tqdm.tqdm(hiers, desc='r')]
        print('mi', sum([p for p in mi if p == 1]), np.median(mi), np.mean(mi), np.min(mi))
        print('ml', sum([p for p in ml if p == 1]), np.median(ml), np.mean(ml), np.min(ml))
        print('to', sum([p for p in to if p == 1]), np.median(to), np.mean(to), np.min(to))
        print('ts', sum([p for p in ts if p == 1]), np.median(ts), np.mean(ts), np.min(ts))
        print('r', sum([p for p in r if p == 1]), np.median(r), np.mean(r), np.min(r))
        data = np.vstack((mi, ml, to, ts, r)).T#, tfh, mlh)).T
        data = pd.DataFrame(np.array(data), columns=['M_I', 'M_L', 'U^1_O', 'U_S', 'R'])
        if csv_buffer: data.to_csv(csv_buffer, index=False)
    return data

def laplacian_analysis(path='lapl_analysis'):
    songs = get_available_songs()#[:5]
    hiers = [buffered_run(DATA+'lapl'+str(i),
        lambda: get_laplacian_struct_from_audio(get_audio(i)))
        for i in tqdm.tqdm(songs, desc='lapl')]
    beats = [get_beats(i) for i in tqdm.tqdm(songs, desc='beats')]
    data = hierarchy_analysis(hiers, beats, 'lapl.csv')
    ax = data.boxplot()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(path+'.pdf', dpi=1000)
    plt.close()

#calculate and make graph of monotonicity and transitivity of salami annotations
def salami_analysis(path='salami_analysis5'):
    songs = get_available_songs()#[:50]
    annos = {i:load_salami_hierarchies(i) for i in songs}
    index = [(i,j) for i in songs
        for j,h in enumerate(load_salami_hierarchies(i))]
    beats = {i:get_beats(i) for i in annos.keys()}
    print("num songs", len(annos))
    hiers = [a for a in flatten(list(annos.values()), 1)]
    beats = flatten([[beats[i] for v in a] for i,a in annos.items()], 1)
    #print(annos[3], hiers[0])
    # print('hiers', len(hiers))
    # print('SILENCE', [h[1][1] for h in hiers if len(np.unique(h[1][1])) == 1])
    # print('uneven', [h[1] for h in hiers if len(np.unique(h[1][1])) < len(np.unique(h[1][0]))])
    #print(interval_monotonicity(hiers[197], beats[197]), label_monotonicity(hiers[197], beats[197]))
    hiers = [homogenize_labels(h) for h in hiers]
    print(sum([1 if auto_labeled(h) else 0 for h in hiers]), len(hiers))
    #print([int_labels(h)[1] for h in hiers if auto_labeled(h)])
    #print(strict_transitivity(hiers[197]), order_transitivity(hiers[197]))
    print(nothing)
    
    #label_count_test(hiers[27], beats[27])
    
    data = hierarchy_analysis(hiers, beats, 'salami.csv')
    
    data.boxplot()
    #transitivity(homogenize_labels(annos[960][0]))
    # print("m1hom", np.mean([monotonicity(h) for h in hiers]))
    # print("m2hom", np.mean([monotonicity2(h, b) for h,b in zip(hiers, beats)]))
    # print("m3hom", np.mean([monotonicity3(h, b) for h,b in zip(hiers, beats)]))
    plot(data.boxplot, path+'.pdf')
    plot(lambda: data.plot.scatter(x='U_S', y='R'), path+'s.pdf')
    plot(lambda: data.plot.scatter(x='M_L', y='R'), path+'s2.pdf')
    plot(lambda: data.plot.scatter(x='M_I', y='M_L'), path+'s3.pdf')
    plot(lambda: data.plot.scatter(x='U_O', y='U_S'), path+'s4.pdf')
    plot(lambda: data.plot.scatter(x='M_I', y='U_S'), path+'s5.pdf')
    plot(lambda: data.plot.scatter(x='M_L', y='U_S'), path+'s6.pdf')
    plot(lambda: data.plot.scatter(x='M_L', y='U_O'), path+'s7.pdf')

def test_mfcc_novelty(index=943):#340 356 (482 574 576)
    ssm = own_chroma_mfcc_affinity(index)[0]
    novelty = get_novelty_boundaries(ssm)
    print(novelty)
    divide_hierarchy

def objective(trial):
    t = trial.suggest_int('t', 0, 0)
    w = trial.suggest_float('w', 0, 0)
    k = trial.suggest_float('k', 2, 4)#, step=0.5)
    m = trial.suggest_int('m', 16, 16)
    s = trial.suggest_float('s', 0, 0)
    #s = trial.suggest_categorical('s', [0.001,0.002,0.004,0.008,0.016,0.032,0.064])
    #k = trial.suggest_float('k', 1, 4, step=1)
    #k = trial.suggest_float('k', 98.25, 99.25)#, step=0.5)
    #k = trial.suggest_int('k', 1, 3, step=5)
    n = trial.suggest_int('n', 100, 100)#, step=50)
    ml = trial.suggest_int('ml', 10, 20)#, step=4)
    md = trial.suggest_int('md', 1, 3, step=1)
    #mg = trial.suggest_int('mg', 6, 8, step=1)
    mg = trial.suggest_int('mg', 0, 0, step=1)
    mgr = trial.suggest_float('mgr', .1, .4)#, step=.1)
    #mgr = trial.suggest_float('mgr', 0.02, 0.05)#, step=.1)
    ml2 = trial.suggest_int('ml2', 10, 20)#, step=4)
    md2 = trial.suggest_int('md2', 1, 1, step=1)
    lex = trial.suggest_int('lex', 1, 1)
    beta = trial.suggest_float('beta', .2, .6)#, step=.1)
    if trial.should_prune():
        raise optuna.TrialPruned()
    #[229, 79, 231, 315, 198] [75, 22, 183, 294, 111]
    #[1270,1461,1375,340,1627,584,1196,443,23,1434] [899,458,811,340,1072,1068,572,310,120,331]
    #[680,95,791,229,1356,236,352,852,384,1168,1132,612,1231,1443,370,794,7,1256,1356,443,1634,791,275,373,332,1098,1186,498,1403,708,1382,616,462,1610,346,578,1266,1654,771,1404,637,344,813,1154,1237,148,618]
    return 100 * comparative_eval([514,1414,386,573,1109,783,1634,845,367,1139,1120,1469,60,690,496,565,1166,818,1077,1372,39,1618,556,320,1213],#[14,244,616,749]))#[340,356,482,574,576],#get_monotonic_salami()[6:100],
        {'MATRIX_TYPE': t, 'MEDIAN_LEN': m, 'SIGMA': s, 'WEIGHT': w, 'THRESHOLD': k,
        'NUM_SEGS': n, 'MIN_LEN': ml, 'MIN_DIST': md, 'MAX_GAPS': mg,
        'MAX_GAP_RATIO': mgr, 'MIN_LEN2': ml2, 'MIN_DIST2': md2, 'LEXIS': lex,
        'BETA': beta})

# conda activate p38
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"

def study():
    study = optuna.create_study(direction='maximize', load_if_exists=True, pruner=RepeatPruner())#, sampler=optuna.samplers.GridSampler())
    # study = optuna.create_study(direction='maximize', load_if_exists=True, pruner=RepeatPruner(),
    #     sampler=optuna.samplers.GridSampler({"s": [0.001,0.002,0.004,0.008,0.016,0.032,0.064],
    #     'beta':[0.2,0.3,0.4,0.5,0.6], 'mgr':[0.2]}))
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    params=['k','ml','md','mgr','ml2','beta']
    ext='16classic.png'
    optuna.visualization.plot_slice(study, params=params).write_image(output+'params'+ext)
    optuna.visualization.plot_param_importances(study, params=params).write_image(output+'pimps'+ext)
    optuna.visualization.plot_optimization_history(study).write_image(output+'poptim'+ext)
    optuna.visualization.plot_contour(study, params=params).write_image(output+'pcont'+ext)

# def sweep(multi=True):
#     #songs = [37,95,107,108,139,148,166,170,192,200]
#     #songs = [37,95,107,108,139,148,166,170,192,200]+get_monotonic_salami()[90:100]#[5:30]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
#     songs = get_available_songs()#get_monotonic_salami()#[6:100]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
#     if multi:
#         multiprocess('evaluating hierarchies', evaluate_to_table, songs, True)
#     else:
#         [evaluate_to_table(i) for i in tqdm.tqdm(songs)]

def calc(i):
    return calculate_fused_matrix(get_audio(i), True)

if __name__ == "__main__":
    #print(np.random.choice(get_available_songs(), 100, replace=False))#[6:100], 5))
    #[880,36,1212,1491,1202,216,1106,1120,1302,715,135,1261,1613,1653,1363,512,1179,1456,376,486,653,340,979,1110,805,1207,983,1454,1630,1127,1479,1038,1069,315,1334,1191,1394,389,1132,613,1307,800,349,356,1112,1054,1311,155,388,1173,587,1291,768,722,261,302,1210,612,626,1379,749,1229,995,1042,1428,1629,1603,1195,327,1388,911,1434,1332,1364,458,735,1242,204,1420,1062,133,1150,1286,1451,330,1048,959,1216,31,1064,374,787,227,751,655,634,1381,509,1282,550]
    #[1630,485,757,790,1207,1607,1019,252,851,1050,1363,632,610,739,1280,770,1200,930,1635,1366,1251,5,228,1061,1314]
    #[613,1275,964,973,386,1138,1056,771,1007,572,376,660,1175,152,1014,282,7,1462,1218,1261,426,1384,293,1293,1003]
    #[1492,618,272,1211,291,1180,904,1162,1331,110,1283,330,1075,682,427,1004,806,531,535,1475,218,884,968,672,156]
    #[399,1402,524,784,805,765,1373,455,1088,316,1190,1438,1462,213,1162,1285,1135,1333,1191,900,1107,1490,231,364,1120]
    #[1099,1179,1210,1431,616,1419,578,749,765,1405,198,1603,1395,1059,696,774,1196,675,1186,1347,458,1648,244,1392,14]
    #[851,1452,362,591,1330,1179,1416,1131,1304,1484,997,367,1356,1188,906,1221,1099,1119,972,852,1080,1498,1077,1326,342,1392,780,735,590,170,1605,1127,363,1415,303,1365,132,426,1100,1158,1006,1036,1223,1607,1274,1333,901,1253,1652,610,1164,1028,323,1044,1360,5,894,1491,944,1187,1427,1132,152,1064,782,312,931,549,861,1084,1160,806,1359,1032,919,632,1307,200,1293,1454,1474,1444,445,1358,686,958,1148,448,604,24,108,690,592,669,909,920,784,402,1266,930]
    #[1330,832,822,944,739,47,48,1362,539,668,383,1394,1156,1232,1072,1435,654,261,1371,658,1086,1400,1191,272,1408]
    #[264,1397,426,158,1352,485,568,1461,1149,819,1084,988,298,1605,959,365,1301,900,1028,31,932,1152,445,768,229]
    #[63,604,1365,608,787,799,655,36,1406,702,1295,1392,1339,611,1234,640,1148,1314,1431,1621,1315,298,1254,1379,108,1174,708,24,146,1206,1176,994,973,1082,1103,213,594,1253,770,1104,1216,974,615,1019,1340,1251,1127,455,607,1349,662,1053,672,1356,1294,992,819,935,582,1284,520,107,668,1156,400,1422,733,1141,7,1306,1210,814,1150,643,1477,925,1647,1355,1399,307,695,1111,762,468,562,227,1028,1203,1290,22,47,95,1382,790,1119,847,597,565,1059,1248]
    #[584,819,781,280,1152,900,726,780,957,1219,1413,51,767,1340,1066,424,515,583,343,1619,307,799,971,1059,632,1445,805,86,527,1351,611,50,702,1188,1341,1070,26,884,615,772,616,1127,346,1299,1388,636,491,536,31,941,1138,1366,1647,906,980,592,1189,520,944,942,1100,790,379,1620,1406,1018,557,1603,1139,994,1259,549,981,1091,914,861,1234,1104,1120,1347,1047,483,1079,1368,1221,1467,968,996,1300,846,578,1042,811,991,1652,1112,1263,671,366,1124,978,1240,740,478,1157,1342,534,63,244,293,1608,404,988,565,1146,1437,572,1359,1296,1256,1274,725,629,1630,1004,532,1110,501,842,1421,236,1422,661,356,1154,405,620,146,973,1615,958,1435,1416,1008,1272,352,11,613,422,439,1148,1275,1253,466,158,482,1099,518,1151,1394,1284,899,984,887,446,152,828,276,1143,472,384,24,294,1355,1364,651,1381,1356,10,1243,464,682,19,331,1247,1408,1136,1474,813,818,1309,261,566,528,1074,1286,460,1382,956,198]
    #[790,1303,998,110,1206,311,451,1264,831,1024,860,14,1448,683,1165,1330,1370,1142,1208,447,53,773,1210,1342,1064]
    #[1383,1051,1292,1371,107,426,686,1322,911,408,1434,733,1492,197,628,1194,1111,1035,1054,1427,1359,1394,1240,493,383,672,82,669,549,1330,1272,1336,1234,1317,1653,508,1363,909,1036,274,548,1023,1490,1022,228,890,1018,607,1447,1290,1196,1459,1399,1380,1301,1242,1222,1378,661,678,1334,1192,1050,108,763,680,1047,1384,1308,955,979,477,638,1263,360,514,1414,386,573,1109,783,1634,845,367,1139,1120,1469,60,690,496,565,1166,818,1077,1372,39,1618,556,320,1213]
    #study()
    #laplacian_analysis()
    #extract_all_features()
    #calculate_fused_matrices()
    #profile(indie_eval)#indie_eval()
    
    #PLOT_PATH=output+'all23/'
    #lapl_eval([14, 'l', PARAMS])
    # # # # #calculate_fused_matrix(get_audio(1210), True)
    #own_eval([14, 't', PARAMS])#578 1059
    #profile(lambda: own_eval([14, 't', PARAMS]))#340
    #matrix_analysis.beta_combi_experiment(1648, PARAMS, PLOT_PATH, RESULTS)#1347 1648
    #matrix_analysis.var_sigma_beta(573, PARAMS, PLOT_PATH)
    
    #multiprocess('fusing matrices', calc, [680,95,791,229,1356,236,352,852,384,1168,1132,612,1231,1443,370,794,7,1256,1356,443,1634,791,275,373,332,1098,1186,498,1403,708,1382,616,462,1610,346,578,1266,1654,771,1404,637,344,813,1154,1237,148,618], True)
    #print(100*comparative_eval([514,1414,386,573,1109,783,1634,845,367,1139,1120,1469,60,690,496,565,1166,818,1077,1372,39,1618,556,320,1213]))#[14,244,616,749]))#[340,356,482,574,576]))
    #wf:-2 wpb:2 2.758309140953134
    #wf:-2 wpb:5 1.044446465454624
    #wf:-2 wpb:1 -1.7147500601471868
    #wf:-1 wpb:2 2.6244372339491586
    
    #plot()
    #matrix_analysis.test_sigma(RESULTS, PARAMS, PLOT_PATH)
    #matrix_analysis.test_beta_combi(RESULTS, PARAMS, PLOT_PATH)
    #matrix_analysis.test_var_sigma_beta(RESULTS, PARAMS, PLOT_PATH)
    #matrix_analysis.test_classic_beta(RESULTS, PARAMS, PLOT_PATH)
    matrix_analysis.test_beta_measure(RESULTS, PARAMS, PLOT_PATH)
    #print(100*comparative_eval([1099,1179,1210,1431,616,1419,578,749,765,1405,198,1603,1395,1059,696,774,1196,675,1186,1347,458,1648,244,1392,14]))#get_available_songs()))
    #multi_eval(get_available_songs(), 'l', lapl_eval, True) #[63,604,1365,608,787,799,655,36,1406,702,1295,1392,1339,611,1234,640,1148,1314,1431,1621,1315,298,1254,1379,108,1174,708,24,146,1206,1176,994,973,1082,1103,213,594,1253,770,1104,1216,974,615,1019,1340,1251,1127,455,607,1349,662,1053,672,1356,1294,992,819,935,582,1284,520,107,668,1156,400,1422,733,1141,7,1306,1210,814,1150,643,1477,925,1647,1355,1399,307,695,1111,762,468,562,227,1028,1203,1290,22,47,95,1382,790,1119,847,597,565,1059,1248])#[408, 822, 722, 637, 527])
    #salami_analysis()
    #run_laplacian([880, 12, None, None])
    #sweep_laplacian([880,36,1212,1491,1202,216,1106,1120,1302,715,135,1261,1613,1653,1363,512,1179,1456,376,486,653,340,979,1110,805,1207,983,1454,1630,1127,1479,1038,1069,315,1334,1191,1394,389,1132,613,1307,800,349,356,1112,1054,1311,155,388,1173,587,1291,768,722,261,302,1210,612,626,1379,749,1229,995,1042,1428,1629,1603,1195,327,1388,911,1434,1332,1364,458,735,1242,204,1420,1062,133,1150,1286,1451,330,1048,959,1216,31,1064,374,787,227,751,655,634,1381,509,1282,550])
    #plot_laplacian()
    #load_salami_hierarchies(197)
    #plot('salamiF9.png')
    #test_mfcc_novelty()
