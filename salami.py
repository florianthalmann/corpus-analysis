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
    segments_to_matrix, get_affinity_matrix, get_segments_from_matrix
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
RESULTS = output+'results.csv'
graphditty = '/Users/flo/Projects/Code/Kyoto/GraphDitty/SongStructure.py'

K_FACTOR = 10
MIN_LEN = 16
MIN_DIST = 1 # >= 1
MAX_GAPS = 6
MAX_GAP_RATIO = .4
MIN_LEN2 = 10
MIN_DIST2 = 1

def get_available_songs():
    return sorted([int(s.split('.')[0]) for s in os.listdir(audio) if '.mp3' in s])

def extract_features(audio):
    extract_chords(audio, features)
    extract_bars(audio, features)

def extract_all_features():
    audio_files = [os.path.join(audio, a) for a in os.listdir(audio)]
    multiprocess('extracting features', extract_features, audio_files)

def calculate_fused_matrix(audio):
    filename = audio.split('/')[-1].replace('.mp3', '')
    subprocess.call(['python', graphditty, '--win_fac', str(-1),
        '--filename', audio, '--matfilename', features+filename+'.mat',
        '--jsonfilename', features+filename+'.json'])#,
        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def calculate_fused_matrices():
    audio_files = [os.path.join(audio, a) for a in os.listdir(audio)]
    multiprocess('calculating fused matrices', calculate_fused_matrix, audio_files)

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

def result_exists(data, columns):
    return (data[data.columns[:len(columns)]] == columns).all(1).any()

def test_hierarchy(index):
    groundtruth = load_salami_hierarchies(index)
    #groundtruth = [homogenize_labels(g) for g in groundtruth]
    #print(groundtruth[0], index)
    exists = False
    if os.path.isfile(RESULTS):
        data = pd.read_csv(RESULTS)
        exists = all([result_exists(data, [index, K_FACTOR, MIN_LEN, MIN_DIST,
            MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2, i, m])
            for i in range(len(groundtruth)) for m in ['transitive']])#, 'laplacian']])
    #if len(groundtruth) > 0 and not exists:
        #print(groundtruth[0])
        maxtime = np.max(np.concatenate(groundtruth[0][0]))
        exists = False
        chroma = get_beatwise_chroma(index)
        #chords = load_beatwise_chords(index)
        #MAX_GAPS, MAX_GAP_RATIO
        affinity, raw = get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO)#, k_factor=K_FACTOR)
        plot_matrix(raw, 'est0.png')
        plot_matrix(affinity, 'est1.png')
        
        # plot_matrix(get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO, 10)[0], 'aff10.png')
        # plot_matrix(get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO, 5)[0], 'aff5.png')
        # plot_matrix(get_affinity_matrix(chroma, chroma, False, MAX_GAPS, MAX_GAP_RATIO, 20)[0], 'aff20.png')
        #SEG_COUNT, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO
        
        # alignment = get_alignment_segments(chroma, chroma, 0,
        #     MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)#, k_factor=K_FACTOR)
        # plot_matrix(segments_to_matrix(alignment, (len(chroma),len(chroma))), 'est2.png')
        # hierarchy = simple_structure(chroma, alignment, MIN_LEN2, MIN_DIST2)
        # beats = get_beats(index)
        
        #matrix, beats = get_smooth_affinity_matrix(get_audio(index))
        matrix, beats = load_fused_matrix(index)
        beats = beats[:len(matrix)]
        alignment = get_segments_from_matrix(matrix, True, 0, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO)
        plot_matrix(segments_to_matrix(alignment, (len(matrix),len(matrix))), 'est2.png')
        hierarchy = simple_structure(matrix[0], alignment, MIN_LEN2, MIN_DIST2)
        
        #alignment2 = get_alignment_segments(chords, chords, 0, 10, 1, 4, 1)
        #plot_matrix(segments_to_matrix(alignment2, (len(chords),len(chords))), 'sall3.png')
        
        beat_ints = np.dstack((beats, np.append(beats[1:], maxtime)))[0]
        
        hi, hl = [beat_ints for h in range(len(hierarchy))], hierarchy.tolist()
        #hi, hl = hi, hl
        #print(len(hi[0]), len(hl[0]))
        lpi, lpl = get_laplacian_struct_from_audio(get_audio(index))
        #lpi, lpl = get_laplacian_struct_from_affinity2(matrix, beats)
        # 
        # #print(homogenize_labels(groundtruth[0]))
        
        # print(type(beats))
        # print(beats[:10])
        # framesize = 0.1
        numframes = 2000#int(maxtime/0.1)
        frames = np.linspace(0, int(maxtime), numframes, endpoint=False)
        # print(type(frames))
        # print(frames[:10])
        hlabels = beatwise((hi, hl), frames)
        llabels = beatwise((lpi, lpl), frames)
        g0labels = beatwise(homogenize_labels(groundtruth[0]), frames)
        plot_sequences(hlabels, 'salami/16 8/'+str(index)+'t.png')
        plot_sequences(llabels, 'salami/16 8/'+str(index)+'l.png')
        plot_sequences(g0labels, 'salami/16 8/'+str(index)+'a1.png')
        if len(groundtruth) > 1:
            g1labels = beatwise(homogenize_labels(groundtruth[1]), frames)
            plot_sequences(g1labels, 'salami/16 8/'+str(index)+'a2.png')
        # 
        # save_json('hlabels.json', hlabels.tolist())
        # save_json('g0labels.json', g0labels.tolist())
        # print(lmeasure(hlabels, g0labels))
        # print(lmeasure(llabels, g0labels))
        
        new_results = []
        print('EVAL', index)
        for i, (refint, reflab) in enumerate(groundtruth):
            #print('EVAL T', index)
            transitive = evaluate_hierarchy(refint, reflab, hi, hl)
            #print('EVAL L', index)
            laplacian = evaluate_hierarchy(refint, reflab, lpi, lpl)
            new_results.append([index,
                K_FACTOR, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2,
                i, 'transitive', transitive[0], transitive[1], transitive[2]])
            new_results.append([index,
                K_FACTOR, MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2,
                i, 'laplacian', laplacian[0], laplacian[1], laplacian[2]])
        #print(transitive, laplacian)
        print(new_results)
        new_results = pd.DataFrame(np.array(new_results),
            columns=['SONG', 'K_FACTOR', 'MIN_LEN', 'MIN_DIST', 'MAX_GAPS',
                'MAX_GAP_RATIO', 'MIN_LEN2', 'MIN_DIST2',
                'REF', 'METHOD', 'P', 'R', 'L'])
        if not os.path.isfile(RESULTS):
            new_results.to_csv(RESULTS, index=False)
        else:
            data = pd.read_csv(RESULTS)
            data = data.append(new_results, ignore_index=True)
            data.to_csv(RESULTS, index=False)
        #return new_results
    
def run():
    result = buffered_run(output+'eval',
        lambda: multiprocess('evaluating hierarchies', test_hierarchy,
        get_available_songs()),
        [MIN_LEN, MIN_DIST, MAX_GAPS, MAX_GAP_RATIO, MIN_LEN2, MIN_DIST2])

def sweep():
    multiprocess('evaluating hierarchies', test_hierarchy,
        get_available_songs()[297:347])#[6:16])
    #print([mean([ r[0]] for r in result])

def plot():
    data = pd.read_csv(RESULTS)
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
test_hierarchy(958)#982)
#load_fused_matrix(1319)
#calculate_fused_matrices()
#test_hierarchy(INDEX)
#plot()
#print(beatwise(homogenize_labels(load_salami_hierarchies(957)[0]), get_beats(957)))
#print(load_salami_hierarchies(972))
#load_salami_hierarchy(1003, 1)
#extract_all_features()