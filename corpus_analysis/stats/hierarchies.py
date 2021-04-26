import numpy as np
from ..features import to_multinomial

def monotonicity(hierarchy):
    ivls = hierarchy[0]
    return all([set(np.unique(ivls[i])) >= set(np.unique(ivls[i-1]))
        for i in range(1, len(ivls))])

#mcfee/kinnard monotonicity
def monotonicity2(hierarchy, beats):
    return pairwise_recalls(beatwise_ints(hierarchy, beats))

#monotonicity without dependencies between different same-label areas
def monotonicity3(hierarchy, beats):
    labels = beatwise_ints(hierarchy, beats)
    labels = np.array([relabel_adjacent(l) for l in labels])
    return pairwise_recalls(labels)

def pairwise_recalls(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    return np.mean([len(same[i].intersection(same[i-1])) / len(same[i])
        for i in range(1, len(same))])

def beatwise_ints(hierarchy, beats):
    hierarchy = to_int_labels(hierarchy)
    hierarchy = [s for s in hierarchy[0] if len(s) > 0],\
        [[int(i) for i in s] for s in hierarchy[1] if len(s) > 0]
    beat_intervals = list(zip(beats[:-1], beats[1:]))
    values = []
    for intervals, labels in zip(*hierarchy):
        values.append([])
        for b in beat_intervals:
            indices = np.where(b[0] >= intervals[:,0])[0]
            values[-1].append(labels[indices[-1]] if len(indices) > 0 else -1)
    return np.array(values)

def to_int_labels(hierarchy):
    unique = np.unique(np.concatenate(hierarchy[1]), axis=0)
    unique_index = lambda f: np.where(unique == f)[0][0]
    labels = [np.array([unique_index(f) for f in s]) for s in hierarchy[1]]
    return hierarchy[0], labels

def relabel_adjacent(labels):
    sp = np.split(labels, np.where(np.diff(labels) != 0)[0]+1)
    return np.hstack([np.repeat(i,len(s)) for i,s in enumerate(sp)])
