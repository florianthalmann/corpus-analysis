import json, csv, math
import numpy as np

def get_beats(path):
    with open(path) as f:
        beats = list(csv.reader(f, delimiter='\t'))
    return [float(b[0]) for b in beats]

def get_chords(path):
    with open(path) as f:
        return json.load(f)[0]

def goIndexToPCSet(index):
  root = int(index%12);
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
    return [feature[m][1] for m in modes]

def get_beatwise_chords(beatsFile, chordsFile):
    beats = get_beats(beatsFile)
    chords = get_chords(chordsFile)
    return [goIndexToPCSet(m) for m in summarize(chords, beats)]
