import os, csv, math, subprocess, json, librosa
from itertools import repeat
from collections import OrderedDict
import numpy as np
from scipy.stats import vonmises, rv_histogram
from madmom.features import RNNBeatProcessor, DBNBeatTrackingProcessor, CRFBeatDetectionProcessor
from .util import load_json, flatten

def extract_essentia(path, outpath):
    if not os.path.isfile(outpath):
        print('extracting essentia for '+path)
        #essentia_streaming_extractor_music
        subprocess.call(['essentia_streaming_extractor_freesound', path, outpath],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_essentia(path):
    return load_json(path)

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

def extract_bars(path, outpath=None, use_librosa=False):
    if not outpath: outpath = '/'.join(path.split('/')[:-1])
    audioFile = '/'.join(path.split('/')[-1:])
    outFile = outpath + '.'.join(audioFile.split('.')[:-1])+'_bars.txt'
    if not os.path.isfile(outFile):
        if use_librosa:
            y, sr = librosa.load(path)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr,
                trim=False, units='time')
            with open(outFile, 'w') as f:
                f.write('\n'.join([str(b) for b in beats]))
        else:
            subprocess.call('DBNDownbeatTracker single -o "'
                +outFile+'" "'+path+'"', shell=True)

def extract_beats(audio_and_outfile):
    audio, outfile = audio_and_outfile
    if not os.path.isfile(outfile):
        proc = DBNBeatTrackingProcessor(fps=100, transition_lambda=2000, min_bpm=55, max_bpm=180)
        #proc = CRFBeatDetectionProcessor(fps=100, min_bpm=55, max_bpm=180)
        beats = proc(RNNBeatProcessor()(audio))
        with open(outfile, 'w') as f:
            f.write('\n'.join([str(b) for b in beats]))

def extract_onsets(audio_and_outfile):
    audio, outfile = audio_and_outfile
    if not os.path.isfile(outfile):
        subprocess.call('CNNOnsetDetector single -o "'
            +outfile+'" "'+audio+'"', shell=True)

def extract_chroma(audio_and_outfile):
    extract_librosa_feature(*audio_and_outfile, librosa.feature.chroma_cqt)

def extract_mfcc(audio_and_outfile):
    extract_librosa_feature(*audio_and_outfile, librosa.feature.mfcc)

def extract_librosa_feature(audio, outfile, func):
    if not os.path.isfile(outfile):
        y, sr = librosa.load(audio)
        feature = func(y, sr)
        np.save(outfile, feature)

def load_beats(path):
    return np.array([float(b[0]) for b in load_madmom_csv(path)])

def load_bars(path):
    return np.array([float(b[0]) for b in load_madmom_csv(path) if int(b[1]) == 1])

def load_onsets(path):
    o = np.array([float(o[0]) for o in load_madmom_csv(path)])
    #workaround for onset extractor returning empty sequence... (only outliers anyway)
    if len(o) == 0:
        return load_beats(path)
    return o

def load_feature(path):
    return np.load(path)

def load_madmom_csv(path):
    with open(path) as f:
        return list(csv.reader(f, delimiter='\t'))

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
    return np.concatenate((np.dstack((timepoints[:-1], timepoints[1:]))[0],
        [[timepoints[-1], np.inf]]))

#returns the durations for which each i in intervals overlaps with interval
#(negative numbers can be ignored)
def get_overlaps(interval, intervals):
    interval = np.tile(interval, (len(intervals), 1))
    #subtract min endpoints from max startpoints for each i in intervals
    return np.min(np.vstack((interval[:,1], intervals[:,1])), axis=0)\
        - np.max(np.vstack((interval[:,0], intervals[:,0])), axis=0)
    #return [min(i[1], interval[1]) - max(i[0], interval[0]) for i in intervals]

def summarize(feature, timepoints):
    t_intervals = to_intervals(np.array(timepoints))
    f_intervals = to_intervals(np.array(feature)[:,0])
    #for each time interval take the index of feature that overlaps the most
    modes = [np.argmax(get_overlaps(t, f_intervals)) for t in t_intervals]
    #return sequence of feature values for the modes
    return np.array([feature[m][1] for m in modes], dtype=int)

#beats param should be list of positions in seconds
def get_beat_summary(feature, beatsFile, beats, srate=22050, fsize=512):
    if beatsFile and beats is None: beats = load_beats(beatsFile)
    beats = np.array(np.around(beats*(srate/fsize)), dtype=int)#to frames
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
    times = load_bars(beatsFile) if bars else load_beats(beatsFile)
    return get_summarized_chords2(times, chordsFile)

def get_summarized_chroma(audioFile, beatsFile=None, beats=None):
    y, sr = librosa.load(audioFile)
    chroma = librosa.feature.chroma_cqt(y, sr)
    return get_beat_summary(chroma, beatsFile, beats, srate=sr).T

def get_summarized_mfcc(audioFile, beatsFile=None, beats=None):
    y, sr = librosa.load(audioFile)
    mfcc = librosa.feature.mfcc(y, sr)
    return get_beat_summary(mfcc, beatsFile, beats, srate=sr).T

def get_summarized_feature(audioFile, featureFile, beatsFile=None, beats=None):
    y, sr = librosa.load(audioFile)#should have saved this along with chroma
    feature = load_feature(featureFile)
    return get_beat_summary(feature, beatsFile, beats, srate=sr).T

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

def tonal_complexity_cf(chroma):#weiss2014quantifying
    fifths = chroma[np.array([0,7,2,9,4,11,6,1,8,3,10,5])]
    fifths = fifths/np.sum(fifths) #normalize
    dft = fifths * np.exp((2*math.pi*1j*np.arange(12))/12)
    return math.sqrt(1 - np.abs(np.mean(dft)))

def tonal_complexity_cf2(chroma):
    fifths = chroma[np.array([0,7,2,9,4,11,6,1,8,3,10,5])]
    fifths = fifths/np.sum(fifths) #normalize
    bins = np.arange(-6, 6)*math.pi/6
    data = np.hstack([np.repeat(b, 1000*f) for f,b in zip(fifths, bins)])
    return 1 - min(1, vonmises.fit(data, fscale=1)[0]/10)

#print(get_labels([[0,0],[1,2],[0,0],[5,5],[1,2],[0,0]]))
#print(label_to_go_index("A"))
#print(summarize([[0,3],[0.5,5],[2.5,7],[3.5,8],[7,9]], [0,1,2,3,4]))
#print(tonal_complexity_cf2(np.array([1.1,0,0.9,0,0,0,0,0.666,0,0,0,0])))