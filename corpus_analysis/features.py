import os, csv, math, subprocess, json, librosa
from itertools import repeat
from collections import OrderedDict
import numpy as np
from .util import load_json, flatten

def extract_essentia(path, outpath):
    if not os.path.isfile(outpath):
        print('extracting essentia for '+path)
        #essentia_streaming_extractor_music
        subprocess.call(['essentia_streaming_extractor_freesound', path, outpath],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_chords(path, outpath=None):
    audioFile = '/'.join(path.split('/')[-1:])
    audioPath = '/'.join(path.split('/')[:-1])
    if not outpath: outpath = audioPath
    outFile = outpath + '.'.join(audioFile.split('.')[:-1])+'_chords.json'
    if not os.path.isfile(outFile):
        pipe = subprocess.Popen(('echo', '-n', '/srv/'+audioFile), stdout=subprocess.PIPE)
        subprocess.call(['docker run --rm -i -v "'+audioPath+':/srv"'
            +' audiocommons/faas-confident-chord-estimator python3 index.py > "'
            +outFile+'"'], shell=True, stdin=pipe.stdout)

def extract_bars(path, outpath=None):
    if not outpath: outpath = '/'.join(path.split('/')[:-1])
    audioFile = '/'.join(path.split('/')[-1:])
    outFile = outpath + '.'.join(audioFile.split('.')[:-1])+'_bars.txt'
    if not os.path.isfile(outFile):
        subprocess.call('DBNDownbeatTracker single -o "'+outFile+'" "'+path+'"',
            shell=True)

def load_beats(path):
    with open(path) as f:
        beats = list(csv.reader(f, delimiter='\t'))
    return np.array([float(b[0]) for b in beats])

def load_bars(path):
    with open(path) as f:
        beats = list(csv.reader(f, delimiter='\t'))
    return [float(b[0]) for b in beats if int(b[1]) == 1]

def get_pitch_class(label):
    pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[label[0]]
    if len(label) > 1 and label[1] == 'b': pc -= 1
    if len(label) > 1 and label[1] == '#': pc += 1
    return pc

def get_triad_quality(label):
    if label.find('dim') >= 0: return 3
    elif label.find('aug') >= 0: return 2
    elif label.find('m') >= 0: return 1
    return 0

def label_to_go_index(label):
    return get_pitch_class(label) + math.floor(12 * get_triad_quality(label))

def go_index_to_label(index):
    name = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B'][int(index%12)]
    type = ['', 'm', 'aug', 'dim'][math.floor(index/12)]
    return name+type

def go_index_to_pcset(index):
  root = int(index%12)
  type = math.floor(index/12)
  pcset = [root, root+4, root+7] if type == 0 else \
    [root, root+3, root+7] if type == 1 else \
    [root, root+4, root+8] if type == 2 else \
    [root, root+3, root+6]
  pcset = [p%12 for p in pcset]
  return sorted(pcset)

def to_intervals(timepoints):
    return list(zip(timepoints, timepoints[1:])) \
        + [(timepoints[-1], float("inf"))]

def get_overlaps(interval, intervals):
    return [min(i[1], interval[1]) - max(i[0], interval[0]) for i in intervals]

def summarize(feature, timepoints):
    t_intervals = to_intervals(timepoints)
    f_intervals = to_intervals([f[0] for f in feature])
    modes = [np.argmax(get_overlaps(t, f_intervals)) for t in t_intervals]
    return np.array([feature[m][1] for m in modes], dtype=int)

def get_beat_summary(feature, beatsFile, srate=22050, fsize=512):
    beats = np.array(np.around(load_beats(beatsFile)*(srate/fsize)), dtype=int)
    return librosa.util.sync(feature, beats)[:,1:]

def load_chords(chordsFile):
    chords = load_json(chordsFile)
    if "chordSequence" in chords:
        return [[c["start"], label_to_go_index(c["label"])]
            for c in chords["chordSequence"]]
    else: return chords[0]

def get_summarized_chords2(beat_times, chordsFile):
    chords = load_chords(chordsFile)
    return summarize(chords, beat_times)

def get_summarized_chords(beatsFile, chordsFile, bars=False):
    time = load_bars(beatsFile) if bars else load_beats(beatsFile)
    chords = load_chords(chordsFile)
    return summarize(chords, time)

def get_duration(chordsFile):
    return load_json(chordsFile)["duration"]

def get_summarized_chroma(audioFile, beatsFile):
    y, sr = librosa.load(audioFile)
    chroma = librosa.feature.chroma_cqt(y, sr)
    return get_beat_summary(chroma, beatsFile, sr).T

def get_summarized_mfcc(audioFile, beatsFile):
    y, sr = librosa.load(audioFile)
    mfcc = librosa.feature.mfcc(librosa.load(audioFile))
    return get_beat_summary(chroma, beatsFile, sr).T

def to_multinomial(sequences):
    unique = np.unique(np.concatenate(sequences), axis=0)
    unique_index = lambda f: np.where(np.all(unique == f, axis=1))[0][0]
    return [np.array([unique_index(f) for f in s]) for s in sequences]

def get_labels(array):
    lens = [len(flatten(a)) for a in array]
    array = [str(a) for a in array]
    unique = list(OrderedDict.fromkeys(array))
    return flatten([list(repeat(unique.index(a), lens[i]))
        for i,a in enumerate(array)])

def to_hierarchy_labels(leadsheet):
    labels = []
    while any(isinstance(s, list) for s in leadsheet):
        labels.append(get_labels(leadsheet))
        leadsheet = flatten(leadsheet, 1)
    labels.append([label_to_go_index(b) for b in leadsheet]) #add raw chords
    labels.insert(0, list(repeat(0, len(labels[0]))))
    return np.array(labels)

def parse_section(name, leadsheet):
    value = leadsheet[name.replace('.','')]
    if isinstance(value, str):
        return parse_section(value, leadsheet)
    elif isinstance(value, list):
        return [parse_section(v, leadsheet) if isinstance(v, str) else v for v in value]
    return value

def load_leadsheets(path, songs):
    leadsheets = []
    for s in songs:
        with open(os.path.join(path, s+'.json')) as f:
            l = json.load(f)
            leadsheets.append(parse_section('_form', l))
    return [to_hierarchy_labels(l) for l in leadsheets]

#print(get_labels([[0,0],[1,2],[0,0],[5,5],[1,2],[0,0]]))
#print(label_to_go_index("A"))