import os, mir_eval, subprocess, tqdm, gc, optuna
from multiprocessing import Pool, cpu_count
from mutagen.mp3 import MP3
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from matplotlib import pyplot as plt
from corpus_analysis.util import multiprocess, plot_matrix, buffered_run,\
    plot_sequences, save_json, load_json, flatten, catch, RepeatPruner, profile
from corpus_analysis.features import extract_chords, extract_bars,\
    get_summarized_chords, get_summarized_chroma, load_beats, get_summarized_mfcc
from corpus_analysis.alignment.affinity import get_alignment_segments,\
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix,\
    matrix_to_segments, threshold_matrix, get_best_segments, ssm, double_smooth_matrix
from corpus_analysis.structure.structure import simple_structure
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity2,\
    to_levels, get_laplacian_struct_from_audio, get_smooth_affinity_matrix
from corpus_analysis.structure.eval import evaluate_hierarchy, simplify
from corpus_analysis.stats.hierarchies import monotonicity, label_monotonicity,\
    interval_monotonicity, beatwise_ints, strict_transitivity, order_transitivity,\
    relabel_adjacent, to_int_labels, auto_labeled, repetitiveness, complexity
from corpus_analysis.structure.novelty import get_novelty_boundaries
from corpus_analysis.data import Data

PARAMS = dict([
    ['MATRIX_TYPE', 2],#0=own, 1=mcfee, 2=fused, 3=own2
    ['WEIGHT', 0.5],#how much mfcc: 1 = only
    ['THRESHOLD', 0],
    ['NUM_SEGS', 200],
    ['MIN_LEN', 6],
    ['MIN_DIST', 1],
    ['MAX_GAPS', 13],
    ['MAX_GAP_RATIO', .75],
    ['MIN_LEN2', 25],
    ['MIN_DIST2', 1],
    ['LEXIS', 1],
    ['BETA', 0.21]
])
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

def matrix_type():
    index = PARAMS['MATRIX_TYPE']
    return 'own' if index == 0 else 'mcfee' if index == 1 else 'fused'

corpus = '/Users/flo/Projects/Code/Kyoto/SALAMI/'
audio = corpus+'all-audio'#'lma-audio/'
annotations = corpus+'salami-data-public/annotations/'
features = corpus+'features/'
output = 'salami/'
DATA = output+'data/'
#RESULTS = Data(None,
RESULTS = Data(output+'resultsF15.csv',
    columns=['SONG']+list(PARAMS.keys())+['REF', 'METHOD', 'P', 'R', 'L'])
# RESULTS = Data(output+'lapl.csv',
#     columns=['SONG', 'LEVELS', 'REF', 'P', 'R', 'L'])
PLOT_PATH=''#output+'all11/'
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

def calculate_fused_matrix(audio):
    filename = audio.split('/')[-1].replace('.mp3', '')
    if not os.path.isfile(features+filename+'.mat')\
            or not os.path.isfile(features+filename+'.json'):
        subprocess.call(['python', graphditty, '--win_fac', str(-1),
            '--filename', audio, '--matfilename', features+filename+'.mat',
            '--jsonfilename', features+filename+'.json'])#,
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

def load_fused_matrix(index, threshold=True):
    m = sio.loadmat(features+str(index)+'.mat')
    j = load_json(features+str(index)+'.json')
    m = np.array(m['Ws']['Fused MFCC/Chroma'][0][0])#['Fused'][0][0])
    if threshold:
        m = threshold_matrix(m, PARAMS['THRESHOLD'])
    m = np.logical_or(m.T, m)#thresholding may lead to asymmetries (peak picking..)
    m = np.triu(m, k=1)#now symmetrix so only keep upper triangle
    beats = np.array(j['times'][:len(m)])
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

def own_chroma_affinity(index, beats=None):
    chroma = beatwise_feature(index, 'chroma', get_beatwise_chroma, beats)
    return own_affinity(index, [chroma], [1], beats)

def own_mfcc_affinity(index, beats=None):
    mfcc = beatwise_feature(index, 'mfcc', get_beatwise_mfcc, beats)
    return own_affinity(index, [mfcc], [1], beats)

def own_chroma_mfcc_affinity(index, beats=None):
    chroma = beatwise_feature(index, 'chroma', get_beatwise_chroma, beats)
    mfcc = beatwise_feature(index, 'mfcc', get_beatwise_mfcc, beats)
    return own_affinity(index, [chroma, mfcc],
        [1-PARAMS['WEIGHT'], PARAMS['WEIGHT']], beats)

def beatwise_feature(index, name, func, beats=None):
    return buffered_run(DATA+name+str(index), lambda: func(index),
        [len(beats)] if beats is not None else [])#load_beatwise_chords(index)

#features is an array of arrays of feature vectors (e.g. chroma, mfcc,...)
def own_affinity(index, features, weights, beats=None):
    features = [w * MinMaxScaler().fit_transform(f)
        for f,w in zip(features, weights)]
    mix = np.hstack(features) if len(features) > 1 else features[0]
    matrix, raw = get_affinity_matrix(mix, mix, False, PARAMS['MAX_GAPS'],
        PARAMS['MAX_GAP_RATIO'], PARAMS['THRESHOLD'])
    if beats is None: beats = get_beats(index)
    return matrix, raw, beats

def own_chroma_affinity_new(index):
    chroma = buffered_run(DATA+'chroma'+str(index),
        lambda: get_beatwise_chroma(index))#load_beatwise_chords(index)
    chroma = MinMaxScaler().fit_transform(chroma)
    
    if PARAMS['MATRIX_TYPE'] == 2:
        matrix, beats = load_fused_matrix(index, False)
    else:
        matrix, beats = ssm(chroma, chroma), get_beats(index)
        
    raw = threshold_matrix(matrix, 1)
    matrix = get_best_segments(matrix, PARAMS['MIN_LEN'],
        min_dist=PARAMS['MIN_DIST'], threshold=PARAMS['THRESHOLD'], len_emph=PARAMS['MAX_GAP_RATIO'])
    
    return matrix, raw, beats

def transitive_hierarchy(matrix, unsmoothed, target, beats, groundtruth, index, plot_file):
    alignment = get_segments_from_matrix(matrix, True, PARAMS['NUM_SEGS'],
        PARAMS['MIN_LEN'], PARAMS['MIN_DIST'], PARAMS['MAX_GAPS'],
        PARAMS['MAX_GAP_RATIO'], unsmoothed)
    
    if PLOT_PATH: plot_matrix(segments_to_matrix(alignment, matrix.shape), PLOT_PATH+str(index)+'-m1f2.png')
    #alignment = sorted(matrix_to_segments(matrix), key=lambda s: len(s), reverse=True)#[:PARAMS['NUM_SEGS']]
    # #TODO STANDARDIZE THIS!!
    # if len(alignment) < 10:
    #     print('alternative matrix!')
    #     matrix, raw, beats = own_chroma_affinity(index, 3)
    #     alignment = get_segments_from_matrix(matrix, True, 100, int(MIN_LEN/2),
    #         MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, raw)
    #     # print(len(alignment))
    #     # plot_matrix(raw, 'm0.png')
    #     # plot_matrix(matrix, 'm1.png')
    #matrix = segments_to_matrix(alignment, matrix.shape)
    #seq = matrix[0] if matrix is not None else []
    #target = unsmoothed#np.where(matrix+unsmoothed > 0, 1, 0)
    hierarchy = simple_structure(alignment, PARAMS['MIN_LEN2'],
        PARAMS['MIN_DIST2'], PARAMS['BETA'], target, lexis=PARAMS['LEXIS'] == 1,
        plot_file=plot_file)
    maxtime = np.max(np.concatenate(groundtruth[0][0]))
    beats = beats[:target.shape[0]]#just to make sure
    beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
    return [beat_ints for h in range(len(hierarchy))], hierarchy.tolist()

def get_groundtruth(index):
    groundtruth = load_salami_hierarchies(index)
    if HOM_LABELS: groundtruth = [homogenize_labels(v) for v in groundtruth]
    if PLOT_PATH: plot_groundtruths(groundtruth, index, PLOT_PATH)
    return groundtruth

def get_hierarchy(index, hierarchy_buffer=None):
    groundtruth = get_groundtruth(index)
    if PARAMS['MATRIX_TYPE'] is 2:
        matrix, beats = load_fused_matrix(index)
        raw = matrix.copy()
        if PLOT_PATH: plot_matrix(matrix, PLOT_PATH+str(index)+'-m0f.png')
        matrix = double_smooth_matrix(matrix, True, PARAMS['MAX_GAPS'],
            PARAMS['MAX_GAP_RATIO'])
        if PLOT_PATH: plot_matrix(matrix, PLOT_PATH+str(index)+'-m1f.png')
    elif PARAMS['MATRIX_TYPE'] is 1:
        matrix, beats = buffered_run(DATA+'mcfee'+str(index),
            lambda: get_smooth_affinity_matrix(get_audio(index)))
        if PLOT_PATH: plot_matrix(matrix, PLOT_PATH+str(index)+'-m1m.png')
    else:
        matrix, raw, beats = buffered_run(DATA+'ownM'+str(index),
            lambda: own_chroma_mfcc_affinity(index), PARAMS.values())
        if PLOT_PATH: plot_matrix(raw, PLOT_PATH+str(index)+'-m0o.png')
        if PLOT_PATH: plot_matrix(matrix, PLOT_PATH+str(index)+'-m1o.png')
    # matrix, raw, beats = own_chroma_affinity_new(index)
    # if PLOT_PATH: plot_matrix(raw, PLOT_PATH+str(index)+'-m0oN.png')
    # if PLOT_PATH: plot_matrix(matrix, PLOT_PATH+str(index)+'-m1oN.png')
    # #raw = None
    
    target = raw#matrix
    #target = load_fused_matrix(index, True)[0]
    #target = (raw-np.min(raw))/(np.max(raw)-np.min(raw))
    # target = buffered_run(DATA+'own'+str(index),
    #     lambda: own_chroma_affinity(index, beats), PARAMS.values())[1]
    # target = np.pad(target, (0, len(beats)-len(target)))
    
    plot_file = PLOT_PATH+str(index)+'-m2'+matrix_type()[0]+'.png' if PLOT_PATH else None
    if hierarchy_buffer is not None:
        own = buffered_run(DATA+hierarchy_buffer+str(index),
            lambda: transitive_hierarchy(matrix, raw, target, beats, groundtruth, index, plot_file), PARAMS.values())
    else:
        own = transitive_hierarchy(matrix, raw, target, beats, groundtruth, index, plot_file)
    if PLOT_PATH: plot_hierarchy(PLOT_PATH, index, 'o'+matrix_type()[0],
        own[0], own[1], groundtruth, force=True)
    return own

#24 31 32 37 47 56   5,14   95  135 148 166     133     1627    231     618
def own_eval(params):
    global PARAMS
    index, method_name, PARAMS = params
    t = get_hierarchy(index)#, 'ownBUFFER')
    return evaluate(index, method_name, list(PARAMS.values()), t[0], t[1])

def lapl_eval(params):
    index, method_name = params[0], params[1]
    l = buffered_run(DATA+'lapl'+str(index),#+method_name+'-'+str(index),
        lambda: get_laplacian_struct_from_audio(get_audio(index)))
    # num_levels = len(own[0])
    # l = l[0][:num_levels], l[1][:num_levels]
    if PLOT_PATH: plot_hierarchy(PLOT_PATH, index, method_name, l[0], l[1], get_groundtruth(index))
    return evaluate(index, method_name, [0 for v in list(PARAMS.values())], l[0], l[1])

def get_ref_rows(index, method_name, ignore_params):
    params = list(PARAMS.values())
    if ignore_params: params = [0 for v in params]
    return [[index]+params+[i, method_name]
        for i in range(len(load_salami_hierarchies(index)))]

def get_missing_indices(indices, method_name, ignore_params):
    ref_rows = [get_ref_rows(i, method_name, ignore_params) for i in indices]
    return [r[0][0] for r in ref_rows if not RESULTS.rows_exist(r)]

def multi_eval(indices, method_name, method_func, ignore_params, params=PARAMS):
    global PARAMS
    PARAMS = params
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
        for s in multi_eval(indices, 't', own_eval, False, params)]
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
    print(data.groupby(['SONG','METHOD'])[['P','R','L']].mean())
    print(nothing)
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
    plott(data.boxplot, path+'.pdf')
    plott(lambda: data.plot.scatter(x='U_S', y='R'), path+'s.pdf')
    plott(lambda: data.plot.scatter(x='M_L', y='R'), path+'s2.pdf')
    plott(lambda: data.plot.scatter(x='M_I', y='M_L'), path+'s3.pdf')
    plott(lambda: data.plot.scatter(x='U_O', y='U_S'), path+'s4.pdf')
    plott(lambda: data.plot.scatter(x='M_I', y='U_S'), path+'s5.pdf')
    plott(lambda: data.plot.scatter(x='M_L', y='U_S'), path+'s6.pdf')
    plott(lambda: data.plot.scatter(x='M_L', y='U_O'), path+'s7.pdf')

def plott(plot_func, path):
    plot_func()
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()
    plt.close()

def test_mfcc_novelty(index=943):#340 356 (482 574 576)
    ssm = own_chroma_mfcc_affinity(index)[0]
    novelty = get_novelty_boundaries(ssm)
    print(novelty)

def objective(trial):
    t = trial.suggest_int('t', 2, 2, step=1)
    w = trial.suggest_float('w', 0.5, 0.5)
    k = trial.suggest_float('k', 0, 0)#, step=0.5)
    #k = trial.suggest_float('k', 1, 4, step=1)
    #k = trial.suggest_float('k', 98.25, 99.25)#, step=0.5)
    #k = trial.suggest_int('k', 1, 3, step=5)
    n = trial.suggest_int('n', 100, 100)#, step=50)
    ml = trial.suggest_int('ml', 5, 10)#, step=4)
    md = trial.suggest_int('md', 1, 5, step=1)
    #mg = trial.suggest_int('mg', 6, 8, step=1)
    mg = trial.suggest_int('mg', 10, 16, step=1)
    mgr = trial.suggest_float('mgr', .4, .8)#, step=.1)
    #mgr = trial.suggest_float('mgr', 0.02, 0.05)#, step=.1)
    ml2 = trial.suggest_int('ml2', 10, 40)#, step=4)
    md2 = trial.suggest_int('md2', 1, 1, step=1)
    lex = trial.suggest_int('lex', 1, 1)
    beta = trial.suggest_float('beta', .1, .4)#, step=.1)
    if trial.should_prune():
        raise optuna.TrialPruned()
    #[229, 79, 231, 315, 198] [75, 22, 183, 294, 111]
    #[1270,1461,1375,340,1627,584,1196,443,23,1434] [899,458,811,340,1072,1068,572,310,120,331]
    #[680,95,791,229,1356,236,352,852,384,1168,1132,612,1231,1443,370,794,7,1256,1356,443,1634,791,275,373,332,1098,1186,498,1403,708,1382,616,462,1610,346,578,1266,1654,771,1404,637,344,813,1154,1237,148,618]
    return 100 * comparative_eval([1099,1179,1210,1431,616,1419,578,749,765,1405,198,1603,1395,1059,696,774,1196,675,1186,1347,458,1648,244,1392,14],#get_monotonic_salami()[6:100],
        {'MATRIX_TYPE': t, 'WEIGHT': w, 'THRESHOLD': k,
        'NUM_SEGS': n, 'MIN_LEN': ml, 'MIN_DIST': md, 'MAX_GAPS': mg,
        'MAX_GAP_RATIO': mgr, 'MIN_LEN2': ml2, 'MIN_DIST2': md2, 'LEXIS': lex,
        'BETA': beta})

# conda activate p38
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"

def study():
    study = optuna.create_study(direction='maximize', load_if_exists=True, pruner=RepeatPruner())#, sampler=optuna.samplers.GridSampler())
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    params=['mgr','ml','mg','ml2','beta']
    ext='16fusedNEW3.png'
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

if __name__ == "__main__":
    #print(np.random.choice(get_available_songs(), 200, replace=False))#[6:100], 5))
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
    study()
    #laplacian_analysis()
    #extract_all_features()
    #calculate_fused_matrices()
    #sweep()
    #profile(indie_eval)#indie_eval()
    #PLOT_PATH=output+'all11/'
    # own_eval([340, 't', PARAMS])
    #print(100*comparative_eval([340,356,482,574,576]))
    #print(comparative_eval([1099,1179,1210,1431,616,1419,578,749,765,1405,198,1603,1395,1059,696,774,1196,675,1186,1347,458,1648,244,1392,14]))#get_available_songs()))
    #multi_eval(get_available_songs(), 'l', lapl_eval, True) #[63,604,1365,608,787,799,655,36,1406,702,1295,1392,1339,611,1234,640,1148,1314,1431,1621,1315,298,1254,1379,108,1174,708,24,146,1206,1176,994,973,1082,1103,213,594,1253,770,1104,1216,974,615,1019,1340,1251,1127,455,607,1349,662,1053,672,1356,1294,992,819,935,582,1284,520,107,668,1156,400,1422,733,1141,7,1306,1210,814,1150,643,1477,925,1647,1355,1399,307,695,1111,762,468,562,227,1028,1203,1290,22,47,95,1382,790,1119,847,597,565,1059,1248])#[408, 822, 722, 637, 527])
    #salami_analysis()
    #run_laplacian([880, 12, None, None])
    #sweep_laplacian([880,36,1212,1491,1202,216,1106,1120,1302,715,135,1261,1613,1653,1363,512,1179,1456,376,486,653,340,979,1110,805,1207,983,1454,1630,1127,1479,1038,1069,315,1334,1191,1394,389,1132,613,1307,800,349,356,1112,1054,1311,155,388,1173,587,1291,768,722,261,302,1210,612,626,1379,749,1229,995,1042,1428,1629,1603,1195,327,1388,911,1434,1332,1364,458,735,1242,204,1420,1062,133,1150,1286,1451,330,1048,959,1216,31,1064,374,787,227,751,655,634,1381,509,1282,550])
    #plot_laplacian()
    #load_salami_hierarchies(197)
    #plot('salamiF9.png')
    #test_mfcc_novelty()
