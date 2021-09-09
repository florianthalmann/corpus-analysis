import os, mir_eval, subprocess, tqdm, gc
from mutagen.mp3 import MP3
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime
from matplotlib import pyplot as plt
from corpus_analysis.util import multiprocess, plot_matrix, buffered_run,\
    plot_sequences, save_json, load_json, flatten, catch
from corpus_analysis.features import extract_chords, extract_bars,\
    get_summarized_chords, get_summarized_chroma, load_beats
from corpus_analysis.alignment.affinity import get_alignment_segments,\
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix,\
    matrix_to_segments
from corpus_analysis.structure.structure import simple_structure
from corpus_analysis.structure.laplacian import get_laplacian_struct_from_affinity2,\
    to_levels, get_laplacian_struct_from_audio, get_smooth_affinity_matrix
from corpus_analysis.structure.eval import evaluate_hierarchy, simplify
from corpus_analysis.stats.hierarchies import monotonicity, monotonicity2,\
    monotonicity3, beatwise_ints, transitivity
from corpus_analysis.data import Data

corpus = '/Users/flo/Projects/Code/Kyoto/SALAMI/'
audio = corpus+'all-audio'#'lma-audio/'
annotations = corpus+'salami-data-public/annotations/'
features = corpus+'features/'
output = 'salami/'
DATA = output+'data/'
RESULTS = Data(output+'resultsF5.csv',
    columns=['SONG', 'K_FACTOR', 'MIN_LEN', 'MIN_DIST', 'MAX_GAPS',
    'MAX_GAP_RATIO', 'MIN_LEN2', 'MIN_DIST2',
    'REF', 'METHOD', 'P', 'R', 'L'])
graphditty = '/Users/flo/Projects/Code/Kyoto/GraphDitty/SongStructure.py'

K_FACTOR = 10
MIN_LEN = 12
MIN_DIST = 1 # >= 1
MAX_GAPS = 7
MAX_GAP_RATIO = .4
MIN_LEN2 = 8
MIN_DIST2 = 1
PLOT_FRAMES = 2000

PARAMS = [K_FACTOR, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2]

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
    [calculate_fused_matrix(a) for a in tqdm.tqdm(get_audio_files())]

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

def load_fused_matrix(index):
    m = sio.loadmat(features+str(index)+'.mat')
    j = load_json(features+str(index)+'.json')
    m = np.array(m['Ws']['Fused'][0][0])
    m[m < 0.01] = 0
    m[m != 0] = 1
    beats = np.array(j['times'][:len(m)])
    #plot_matrix(m)
    return m, beats

def get_monotonic_salami():
    annos = {i:load_salami_hierarchies(i) for i in get_available_songs()}
    return [i for i in annos.keys() if all([monotonicity(h) for h in annos[i]])]

def plot_hierarchy(path, index, method_name, intervals, labels, groundtruth, force=False):
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

def eval_to_rows(index, method_name, groundtruth, intervals, labels):
    results = []
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        results.append([index]+PARAMS+[i, method_name, score[0], score[1], score[2]])
    print(results)
    return results

def eval_and_add_results(index, method_name, groundtruth, intervals, labels, plot_path=None):
    if plot_path:
        plot_hierarchy(plot_path, index, method_name, intervals, labels, groundtruth)
    ref_rows = [[index]+PARAMS+[i, method_name] for i in range(len(groundtruth))]
    rows_func = lambda: eval_to_rows(index, method_name, groundtruth, intervals, labels)
    RESULTS.add_rows(ref_rows, rows_func)

def own_chroma_affinity(index, factor=1, knn=True):
    chroma = buffered_run(DATA+'chroma'+str(index),
        lambda: get_beatwise_chroma(index))#load_beatwise_chords(index)
    matrix, raw = get_affinity_matrix(chroma, chroma, False, MAX_GAPS,
        MAX_GAP_RATIO, factor, knn=knn)#, k_factor=K_FACTOR)
    beats = get_beats(index)
    #plot_matrix(raw, 'm0.png')
    #plot_matrix(matrix, 'm1.png')
    return matrix, raw, beats

def transitive_hierarchy(matrix, unsmoothed, beats, groundtruth, index):
    alignment = get_segments_from_matrix(matrix, True, 100, MIN_LEN,
        MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, unsmoothed)
    if len(alignment) < 10:
        print('alternative matrix!')
        matrix, unsmoothed, beats = own_chroma_affinity(index, 2, knn=True)
        alignment = get_segments_from_matrix(matrix, True, 100, int(MIN_LEN/2),
            MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, unsmoothed)
    matrix = segments_to_matrix(alignment, (len(matrix), len(matrix)))
    seq = matrix[0] if matrix is not None else []
    target = unsmoothed#np.where(matrix+unsmoothed > 0, 1, 0)
    hierarchy = simple_structure(seq, alignment, MIN_LEN2, MIN_DIST2, target)
    maxtime = np.max(np.concatenate(groundtruth[0][0]))
    beats = beats[:len(matrix)]#just to make sure
    beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
    return [beat_ints for h in range(len(hierarchy))], hierarchy.tolist()

def evaluate(index, hom_labels=False, plot_path=output+'all3/'):
    groundtruth = load_salami_hierarchies(index)
    if hom_labels: groundtruth = [homogenize_labels(v) for v in groundtruth]
    #plot_groundtruths(groundtruth, index, plot_path)
    fused, fbeats = load_fused_matrix(index)
    mcfee, mbeats = buffered_run(DATA+'mcfee'+str(index),
        lambda: get_smooth_affinity_matrix(get_audio(index)))
    own, raw, obeats = buffered_run(DATA+'ownN'+str(index),
        lambda: own_chroma_affinity(index), PARAMS)
    
    l = buffered_run(DATA+'lapl'+str(index),
        lambda: get_laplacian_struct_from_audio(get_audio(index)))
    
    #plot_hierarchy(plot_path, index, 'l', l[0], l[1], groundtruth)
    eval_and_add_results(index, 'l', groundtruth, l[0], l[1])
    
    # l_own = buffered_run(DATA+'l_own'+str(index),
    #     lambda: get_laplacian_struct_from_affinity2(own, obeats), PARAMS)
    # eval_and_add_results(index, 'l_own', groundtruth, l_own[0], l_own[1])
    
    t_own = buffered_run(DATA+'transf.25fu'+str(index),
        lambda: transitive_hierarchy(fused, None, fbeats, groundtruth, index), PARAMS)
    #plot_hierarchy(plot_path, index, 't_own10nn', t_own[0], t_own[1], groundtruth)
    eval_and_add_results(index, 'transf.25fu', groundtruth, t_own[0], t_own[1])
    gc.collect()

#24 31 32 37 47 56   5,14   95
def test_own_eval(index=135, plot_path=output+'all3/'):#22):#32#38):
    groundtruth = load_salami_hierarchies(index)
    plot_groundtruths(groundtruth, index, plot_path)
    l = buffered_run(DATA+'lapl'+str(index),
        lambda: get_laplacian_struct_from_audio(get_audio(index)))
    plot_hierarchy(plot_path, index, 'l', l[0], l[1], groundtruth)
    print('affinity')
    aff, unsmoo, beats = own_chroma_affinity(index)
    aff2, beats = buffered_run(DATA+'mcfee'+str(index),
        lambda: get_smooth_affinity_matrix(get_audio(index)))
    #plot_matrix(aff2, 'm3.png')
    print('hierarchy')
    intervals, labels = transitive_hierarchy(aff, unsmoo, beats, groundtruth, index)
    #plot_hierarchy(output+'all2/', index, 't_own200', intervals, labels, groundtruth)
    #print(l[0][:4])
    plot_hierarchy(plot_path, index, 't_ownf', intervals, labels, groundtruth, True)
    print('eval')
    if len(groundtruth) > 1:
        print(evaluate_hierarchy(*groundtruth[0], *groundtruth[1]))
    # for i in intervals:
    #     print(i[i == 162.74866213])
    #     i[np.round(i) == 14] = 13.44435374
    #     i[np.round(i) == 163] = 166.51029478
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, l[0], l[1])
        print([index]+PARAMS+[i, score[0], score[1], score[2]])
    for i, (refint, reflab) in enumerate(groundtruth):
        score = evaluate_hierarchy(refint, reflab, intervals, labels)
        print([index]+PARAMS+[i, score[0], score[1], score[2]])

def plot(path=None):
    data = RESULTS.get_rows()
    #data = data[1183 <= data['SONG']][data['SONG'] <= 1211]
    #data = data[data['SONG'] <= 333]
    #data = data[data['MIN_LEN'] == 24]
    #data = data[(data['K_FACTOR'] == 5) | (data['K_FACTOR'] == 10)]
    #data.groupby(['METHOD']).mean().T.plot(legend=True)
    #data.groupby(['METHOD']).boxplot(column=['P','R','L'])
    print(data.groupby(['METHOD']).mean())
    print(data[data['METHOD'] == 'l'].groupby(['SONG']).max().groupby(['SONG']).mean())
    print(data[data['METHOD'] == 'trans.9'].groupby(['SONG']).max().groupby(['SONG']).mean())
    #print(data[(data['METHOD'] == 't_ownNY') | (data['METHOD'] == 'l')].sort_values(['SONG', 'METHOD']).to_string())
    #print(data.groupby(['SONG', 'METHOD']).mean().sort_values(['SONG', 'METHOD']).to_string())
    #print(data[data['METHOD'] != 'l'].groupby(['SONG']).mean())
    data.boxplot(column=['P','R','L'], by=['METHOD'])
    #plt.show()
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()

#calculate and make graph of monotonicity and transitivity of salami annotations
def salami_analysis(path='salami_analysis.pdf'):
    annos = {i:load_salami_hierarchies(i) for i in get_available_songs()}
    beats = {i:get_beats(i) for i in annos.keys()}
    print("num songs", len(annos))
    hiers = [a for a in flatten(list(annos.values()), 1)]
    beats = flatten([[beats[i] for v in a] for i,a in annos.items()], 1)
    #print(annos[3], hiers[0])
    #check = lambda i,a,b: [print(i), monotonicity2(a,b)]
    #[[check(i,v,beats[i]) for v in a] for i,a in annos.items()]
    #hiers = hiers[:40]
    #m1 = [monotonicity(h) for h in tqdm.tqdm(hiers, desc='m1')]
    mi = [monotonicity3(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='m3')]
    ml = [monotonicity2(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='m2')]
    tf = [transitivity(h) for h in tqdm.tqdm(hiers, desc='t')]
    hiers = [homogenize_labels(h) for h in hiers]
    mlh = [monotonicity2(h, b) for h,b in tqdm.tqdm(list(zip(hiers, beats)), desc='m2')]
    tfh = [transitivity(h) for h in tqdm.tqdm(hiers, desc='th')]
    
    print('mi', sum([p for p in mi if p == 1]))
    print('ml', sum([p for p in ml if p == 1]))
    print('tf', sum([p for p in tf if p == 1]))
    print('mlh', sum([p for p in mlh if p == 1]))
    print('tfh', sum([p for p in tfh if p == 1]))
    
    data = np.vstack((mi, ml, tfh, tf)).T#, tfh, mlh)).T
    pd.DataFrame(np.array(data), columns=['M_I', 'M_L', 'T_O', 'T_F']).boxplot()
    #transitivity(homogenize_labels(annos[960][0]))
    # print("m1hom", np.mean([monotonicity(h) for h in hiers]))
    # print("m2hom", np.mean([monotonicity2(h, b) for h,b in zip(hiers, beats)]))
    # print("m3hom", np.mean([monotonicity3(h, b) for h,b in zip(hiers, beats)]))
    plt.tight_layout()
    plt.savefig(path, dpi=1000) if path else plt.show()

# conda activate p38
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"

def sweep(multi=True):
    #songs = [37,95,107,108,139,148,166,170,192,200]
    songs = [37,95,107,108,139,148,166,170,192,200]+get_monotonic_salami()[90:100]#[5:30]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
    #songs = get_monotonic_salami()[0:100]#get_available_songs()[:100]#[197:222]#[197:347]#[6:16]
    print(len(songs))
    if multi:
        multiprocess('evaluating hierarchies', evaluate, songs, True)
    else:
        [evaluate(i) for i in tqdm.tqdm(songs)]

if __name__ == "__main__":
    #extract_all_features()
    #calculate_fused_matrices()
    #sweep()
    #test_own_eval()
    #evaluate(1199)#1221)
    #salami_analysis()
    #plot('salamiF2.png')
