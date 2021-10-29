import os, mir_eval, subprocess, tqdm, gc, optuna
from mutagen.mp3 import MP3
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from matplotlib import pyplot as plt
from corpus_analysis.util import multiprocess, plot_matrix, buffered_run,\
    plot_sequences, save_json, load_json, flatten, catch, RepeatPruner
from corpus_analysis.features import extract_chords, extract_bars,\
    get_summarized_chords, get_summarized_chroma, load_beats, get_summarized_mfcc
from corpus_analysis.alignment.affinity import get_alignment_segments,\
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix,\
    matrix_to_segments, threshold_matrix, get_best_segments, ssm
from corpus_analysis.structure.structure import simple_structure
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity2,\
    to_levels, get_laplacian_struct_from_audio, get_smooth_affinity_matrix
from corpus_analysis.structure.eval import evaluate_hierarchy, simplify
from corpus_analysis.stats.hierarchies import monotonicity, label_monotonicity,\
    interval_monotonicity, beatwise_ints, strict_transitivity, order_transitivity
from corpus_analysis.data import Data

PARAMS = dict([
    ['MATRIX_TYPE', 0],#0=own, 1=mcfee, 2=fused, 3=own2
    ['THRESHOLD', 99.5],
    ['NUM_SEGS', 100],
    ['MIN_LEN', 3],
    ['MIN_DIST', 1],
    ['MAX_GAPS', 5],
    ['MAX_GAP_RATIO', .4],
    ['MIN_LEN2', 8],
    ['MIN_DIST2', 1],
    ['LEXIS', 1],
    ['BETA', 0.75]
])

def matrix_type():
    index = PARAMS['MATRIX_TYPE']
    return 'own' if index == 0 else 'mcfee' if index == 1 else 'fused'

corpus = '/Users/flo/Projects/Code/Kyoto/SALAMI/'
audio = corpus+'all-audio'#'lma-audio/'
annotations = corpus+'salami-data-public/annotations/'
features = corpus+'features/'
output = 'salami/'
DATA = output+'data/'
# RESULTS = Data(output+'resultsF8.csv',
#     columns=['SONG']+list(PARAMS.keys())+['REF', 'METHOD', 'P', 'R', 'L'])
RESULTS = Data(output+'lapl.csv',
    columns=['SONG', 'LEVELS', 'REF', 'P', 'R', 'L'])
PLOT_PATH=output+'all8/'
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

def get_beatwise_chroma(index):
    return get_summarized_chroma(get_audio(index),
        features+str(index)+'_bars.txt')

def get_beatwise_mfcc(index):
    return get_summarized_mfcc(get_audio(index),
        features+str(index)+'_bars.txt')

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
    m = np.array(m['Ws']['Fused'][0][0])
    if threshold:
        m = threshold_matrix(m, PARAMS['THRESHOLD'])
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

def evaluate(index, method_name, groundtruth, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index]+list(PARAMS.values())+[i, method_name, score[0], score[1], score[2]])
    print(results)
    return results

def eval_and_add_results(index, method_name, groundtruth, intervals, labels):
    ref_rows = [[index]+list(PARAMS.values())+[i, method_name] for i in range(len(groundtruth))]
    rows_func = lambda: evaluate(index, method_name, groundtruth, intervals, labels)
    return RESULTS.add_rows(ref_rows, rows_func)

def own_chroma_affinity(index):
    chroma = buffered_run(DATA+'chroma'+str(index),
        lambda: get_beatwise_chroma(index))#load_beatwise_chords(index)
    chroma = MinMaxScaler().fit_transform(chroma)
    beats = get_beats(index)
    matrix, raw = get_affinity_matrix(chroma, chroma, False, PARAMS['MAX_GAPS'],
        PARAMS['MAX_GAP_RATIO'], PARAMS['THRESHOLD'])
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
        min_dist=PARAMS['MIN_DIST'], threshold=PARAMS['THRESHOLD'])
    
    return matrix, raw, beats

def own_chroma_mfcc_affinity(index, factor=2):
    chroma = buffered_run(DATA+'chroma'+str(index),
        lambda: get_beatwise_chroma(index))#load_beatwise_chords(index)
    mfcc = buffered_run(DATA+'mfcc'+str(index),
        lambda: get_beatwise_mfcc(index))#load_beatwise_chords(index)
    chroma = MinMaxScaler().fit_transform(chroma)
    mfcc = MinMaxScaler().fit_transform(mfcc)
    mix = np.hstack((chroma, mfcc))
    matrix, raw = get_affinity_matrix(mix, mix, False, PARAMS['MAX_GAPS'],
        PARAMS['MAX_GAP_RATIO'], factor)
    beats = get_beats(index)
    return matrix, raw, beats

def transitive_hierarchy(matrix, unsmoothed, beats, groundtruth, index, plot_file):
    alignment = get_segments_from_matrix(matrix, True, PARAMS['NUM_SEGS'],
        PARAMS['MIN_LEN'], PARAMS['MIN_DIST'], PARAMS['MAX_GAPS'],
        PARAMS['MAX_GAP_RATIO'], unsmoothed)
    # #TODO STANDARDIZE THIS!!
    # if len(alignment) < 10:
    #     print('alternative matrix!')
    #     matrix, raw, beats = own_chroma_affinity(index, 3)
    #     alignment = get_segments_from_matrix(matrix, True, 100, int(MIN_LEN/2),
    #         MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, raw)
    #     # print(len(alignment))
    #     # plot_matrix(raw, 'm0.png')
    #     # plot_matrix(matrix, 'm1.png')
    matrix = segments_to_matrix(alignment, (len(matrix), len(matrix)))
    seq = matrix[0] if matrix is not None else []
    target = unsmoothed#np.where(matrix+unsmoothed > 0, 1, 0)
    hierarchy = simple_structure(seq, alignment, PARAMS['MIN_LEN2'],
        PARAMS['MIN_DIST2'], PARAMS['BETA'], target, lexis=PARAMS['LEXIS'] == 1,
        plot_file=plot_file)
    maxtime = np.max(np.concatenate(groundtruth[0][0]))
    beats = beats[:len(matrix)]#just to make sure
    beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
    return [beat_ints for h in range(len(hierarchy))], hierarchy.tolist()

def get_hierarchies(index, hierarchy_buffer=None, plot_path=None):
    groundtruth = load_salami_hierarchies(index)
    if HOM_LABELS: groundtruth = [homogenize_labels(v) for v in groundtruth]
    if plot_path: plot_groundtruths(groundtruth, index, plot_path)
    # if PARAMS['MATRIX_TYPE'] is 2:
    #     matrix, beats = load_fused_matrix(index)
    #     if plot_path: plot_matrix(matrix, plot_path+str(index)+'-m1f.png')
    # elif PARAMS['MATRIX_TYPE'] is 1:
    #     matrix, beats = buffered_run(DATA+'mcfee'+str(index),
    #         lambda: get_smooth_affinity_matrix(get_audio(index)))
    #     if plot_path: plot_matrix(matrix, plot_path+str(index)+'-m1m.png')
    # else:
    #     matrix, raw, beats = buffered_run(DATA+'own'+str(index),
    #         lambda: own_chroma_affinity(index), PARAMS.values())
    #     if plot_path: plot_matrix(raw, plot_path+str(index)+'-m0o.png')
    #     if plot_path: plot_matrix(matrix, plot_path+str(index)+'-m1o.png')
    matrix, raw, beats = own_chroma_affinity_new(index)
    if plot_path: plot_matrix(raw, plot_path+str(index)+'-m0oN.png')
    if plot_path: plot_matrix(matrix, plot_path+str(index)+'-m1oN.png')
    
    l = buffered_run(DATA+'lapl'+str(index),
            lambda: get_laplacian_struct_from_audio(get_audio(index)))
    if plot_path: plot_hierarchy(plot_path, index, 'l', l[0], l[1], groundtruth)
    plot_file = plot_path+str(index)+'-m2'+matrix_type()[0]+'.png' if plot_path else None
    if hierarchy_buffer is not None:
        own = buffered_run(DATA+hierarchy_buffer+str(index),
            lambda: transitive_hierarchy(matrix, None, beats, groundtruth, index, plot_file), PARAMS.values())
    else:
        own = transitive_hierarchy(matrix, None, beats, groundtruth, index, plot_file)
    if plot_path: plot_hierarchy(plot_path, index, 'o'+matrix_type()[0],
        own[0], own[1], groundtruth, force=True)
    return l, own, groundtruth

def evaluate_to_table(index):
    ref_rows = [[index]+list(PARAMS.values())+[i, METHOD_NAME]
        for i in range(len(load_salami_hierarchies(index)))]
    #if not RESULTS.rows_exist(ref_rows):
    l, own, gt = get_hierarchies(index, METHOD_NAME)
    eval_and_add_results(index, 'l', gt, l[0], l[1])
    eval_and_add_results(index, 't', gt, own[0], own[1])#METHOD_NAME, gt, own[0], own[1])
    # l_own = buffered_run(DATA+'l_own'+str(index),
    #     lambda: get_laplacian_struct_from_affinity2(own, obeats), PARAMS)
    # eval_and_add_results(index, 'l_own', gt, l_own[0], l_own[1])
    gc.collect()

#24 31 32 37 47 56   5,14   95  135 148 166     133     1627    231
def indie_eval(params=[443, PARAMS]):#index=95):#22):#32#38):
    global PARAMS
    index, PARAMS = params
    ref_rows = [[index]+list(PARAMS.values())+[i, METHOD_NAME]
        for i in range(len(load_salami_hierarchies(index)))]
    if not RESULTS.rows_exist(ref_rows):
        l, own, gt = get_hierarchies(index)#, plot_path=PLOT_PATH)
        # #compare groundtruths with each other
        # if len(gt) > 1:
        #     print(evaluate_hierarchy(*gt[0], *gt[1]))
        
        # num_levels = len(own[0])
        # l = l[0][:num_levels], l[1][:num_levels]
        
        #compare laplacian and own to groundtruth
        # evl = evaluate(index, 'l', gt, l[0], l[1])
        # evt = evaluate(index, 't', gt, own[0], own[1])
    
        evl = eval_and_add_results(index, 'l', gt, l[0], l[1])
        evt = eval_and_add_results(index, 't', gt, own[0], own[1])
    else:
        gt = load_salami_hierarchies(index)
        evl = eval_and_add_results(index, 'l', gt, None, None)
        evt = eval_and_add_results(index, 't', gt, None, None)
    return np.mean([e[-1] for e in evl]), np.mean([e[-1] for e in evt])

def multi_eval(indices, params=PARAMS):
    params = [[i, params] for i in indices]
    results = multiprocess('multi eval', indie_eval, params, True)
    print(results, np.mean([r[1]-r[0] for r in results]))
    return np.mean([r[1]-r[0] for r in results])

def eval_laplacian(index, levels, groundtruth, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index, levels, i, score[0], score[1], score[2]])
    print(results)
    return results

def run_laplacian(params):
    index, levels = params
    ref_rows = [[index, levels, i]
        for i in range(len(load_salami_hierarchies(index)))]
    if not RESULTS.rows_exist(ref_rows):
        gt = load_salami_hierarchies(index)
        l = get_laplacian_struct_from_audio(get_audio(index), levels)
        rows_func = lambda: eval_laplacian(index, levels, gt, l[0], l[1])
        return RESULTS.add_rows(ref_rows, rows_func)

def sweep_laplacian(indices):
    for l in range(2, 20):
        params = [[i, l] for i in indices]
        multiprocess('multi eval', run_laplacian, params, True)

def plot(path=None):
    data = RESULTS.read()
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

#calculate and make graph of monotonicity and transitivity of salami annotations
def salami_analysis(path='salami_analysis5.pdf'):
    annos = {i:load_salami_hierarchies(i) for i in get_available_songs()[:]}
    beats = {i:get_beats(i) for i in annos.keys()}
    print("num songs", len(annos))
    hiers = [a for a in flatten(list(annos.values()), 1)]
    beats = flatten([[beats[i] for v in a] for i,a in annos.items()], 1)
    #print(annos[3], hiers[0])
    # print('hiers', len(hiers))
    # print('SILENCE', [h[1][1] for h in hiers if len(np.unique(h[1][1])) == 1])
    # print('uneven', [h[1] for h in hiers if len(np.unique(h[1][1])) < len(np.unique(h[1][0]))])
    #print(interval_monotonicity(hiers[197], beats[197]), label_monotonicity(hiers[197], beats[197]))
    #print(strict_transitivity(hiers[197]), order_transitivity(hiers[197]))
    
    h, b = hiers[27], beats[27]
    print(b[:10])
    means = np.append(np.mean(np.vstack((b[:-1], b[1:])), axis=0), [0])
    print(means[:10])
    beats2 = np.vstack((b,means)).reshape((-1,),order='F')[:-1]
    print(beats2[:10])
    print(label_monotonicity(h, b), label_monotonicity(h, beats2))
    print(nothing)
    #check = lambda i,a,b: [print(i), monotonicity2(a,b)]
    #[[check(i,v,beats[i]) for v in a] for i,a in annos.items()]
    #hiers = hiers[:40]
    #m1 = [monotonicity(h) for h in tqdm.tqdm(hiers, desc='m1')]
    #hiers = [homogenize_labels(h) for h in hiers]
    mi = [interval_monotonicity(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='mi')]
    ml = [label_monotonicity(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='ml')]
    to = [order_transitivity(h) for h in tqdm.tqdm(hiers, desc='to')]
    ts = [strict_transitivity(h) for h in tqdm.tqdm(hiers, desc='ts')]
    # hiers = [homogenize_labels(h) for h in hiers]
    # mih = [monotonicity3(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='m3')]
    # mlh = [monotonicity2(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='m2')]
    # tfh = [transitivity(h) for h in tqdm.tqdm(hiers, desc='th')]
    
    #print([(i,l,hiers[k][1]) for k,(i,l) in enumerate(zip(mi,ml)) if i < l])
    
    print('mi', sum([p for p in mi if p == 1]), np.median(mi), np.mean(mi), np.min(mi))
    print('ml', sum([p for p in ml if p == 1]), np.median(ml), np.mean(ml), np.min(ml))
    print('to', sum([p for p in to if p == 1]), np.median(to), np.mean(to), np.min(to))
    print('ts', sum([p for p in ts if p == 1]), np.median(ts), np.mean(ts), np.min(ts))
    # print('mih', sum([p for p in mlh if p == 1]), np.median(mih))
    # print('mlh', sum([p for p in mlh if p == 1]), np.median(mlh))
    # print('tfh', sum([p for p in tfh if p == 1]), np.median(tfh))
    
    data = np.vstack((mi, ml, to, ts)).T#, tfh, mlh)).T
    pd.DataFrame(np.array(data), columns=['M_I', 'M_L', 'T_O', 'T_S']).boxplot()
    #transitivity(homogenize_labels(annos[960][0]))
    # print("m1hom", np.mean([monotonicity(h) for h in hiers]))
    # print("m2hom", np.mean([monotonicity2(h, b) for h,b in zip(hiers, beats)]))
    # print("m3hom", np.mean([monotonicity3(h, b) for h,b in zip(hiers, beats)]))
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()

def objective(trial):
    t = trial.suggest_int('t', 0, 0, step=1)
    k = trial.suggest_float('k', 96, 99, step=1)
    #k = trial.suggest_int('k', 65, 95, step=5)
    #k = trial.suggest_int('k', 1, 3, step=5)
    n = trial.suggest_int('n', 100, 100, step=20)
    ml = trial.suggest_int('ml', 16, 16, step=4)
    md = trial.suggest_int('md', 1, 1, step=1)
    mg = trial.suggest_int('mg', 5, 5, step=2)
    mgr = trial.suggest_float('mgr', .4, .4, step=.1)
    ml2 = trial.suggest_int('ml2', 8, 8, step=2)
    md2 = trial.suggest_int('md2', 1, 1, step=1)
    lex = trial.suggest_int('lex', 1, 1)
    beta = trial.suggest_float('beta', .75, .75, step=.25)
    if trial.should_prune():
        raise optuna.TrialPruned()
    #[229, 79, 231, 315, 198] [75, 22, 183, 294, 111]
    #[1270,1461,1375,340,1627,584,1196,443,23,1434] [899,458,811,340,1072,1068,572,310,120,331]
    return 100 * multi_eval([680,95,791,229,1356,236,352,852,384,1168,1132,612,1231,1443,370,794,7,1256,1356,443,1634,791,275,373,332,1098,1186,498,1403,708,1382,616,462,1610,346,578,1266,1654,771,1404,637,344,813,1154,1237,148,618],#get_monotonic_salami()[6:100],
        {'MATRIX_TYPE': t, 'THRESHOLD': k,
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
    optuna.visualization.plot_slice(study, params=['k','md','beta']).write_image(output+'params.png')
    optuna.visualization.plot_param_importances(study, params=['k','md','beta']).write_image(output+'pimps.png')
    optuna.visualization.plot_optimization_history(study).write_image(output+'poptim.png')
    optuna.visualization.plot_contour(study, params=['k','md','beta']).write_image(output+'pcont.png')

def sweep(multi=True):
    #songs = [37,95,107,108,139,148,166,170,192,200]
    #songs = [37,95,107,108,139,148,166,170,192,200]+get_monotonic_salami()[90:100]#[5:30]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
    songs = get_available_songs()#get_monotonic_salami()#[6:100]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
    if multi:
        multiprocess('evaluating hierarchies', evaluate_to_table, songs, True)
    else:
        [evaluate_to_table(i) for i in tqdm.tqdm(songs)]

if __name__ == "__main__":
    #print(np.random.choice(get_monotonic_salami(), 100))#[6:100], 5))
    #study()
    #extract_all_features()
    #calculate_fused_matrices()
    #sweep()
    #indie_eval()
    #multi_eval([75, 22, 183, 294, 111])#[408, 822, 722, 637, 527])
    #salami_analysis()
    sweep_laplacian(get_available_songs()[:50])
    #load_salami_hierarchies(197)
    #plot('salamiF8.png')
