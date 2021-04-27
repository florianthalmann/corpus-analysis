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

def transitivity(hierarchy):
    tree = to_tree(hierarchy)
    print(tree_to_label_array(tree))

def pairwise_recalls(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    return np.mean([len(same[i].intersection(same[i-1])) / len(same[i])
        for i in range(1, len(same))])

def to_tree(hierarchy):
    #add top node
    hierarchy = list(hierarchy[0]), list(hierarchy[1])
    if len(hierarchy[0][0]) > 1:
        hierarchy[0].insert(0,
            np.array([[hierarchy[0][0][0][0], hierarchy[0][0][-1][-1]]]))
        hierarchy[1].insert(0, ['S'])
    #make monotonic so that every child node has a unique parent
    hierarchy = make_monotonic(hierarchy)
    #initialize with lowest level
    tree = [(i,l,[]) for i,l in zip(hierarchy[0][-1], hierarchy[1][-1])]
    #iteratively build higher levels
    for intervals,labels in list(reversed(list(zip(*hierarchy))))[1:]:
        for i,l in zip(intervals,labels):
            children = [n for n in tree if i[0] <= n[0][0] and n[0][1] <= i[1]]
            #didn't want to work otherwise
            tree = [n for n in tree if not (i[0] <= n[0][0] and n[0][1] <= i[1])]
            tree.append((i,l,children))
    return tree[0]

def tree_to_label_array(tree):
    return [tree[1]] + [tree_to_array(c) for c in tree[2]]

def make_monotonic(hierarchy):
    parent_times = ()
    output = []
    for intervals, labels in zip(*hierarchy):
        for t in parent_times:
            intervals, labels = split_level_at(t, intervals, labels)
        output.append([intervals, labels])
        parent_times = set(np.unique(intervals))
    return list(zip(*output))

def split_level_at(time, intervals, labels):
    index = next((i for i,v in enumerate(intervals) if v[0] < time < v[1]), -1)
    if index >= 0:
        s1, s2 = [intervals[index][0], time], [time, intervals[index][1]]
        intervals = np.insert(intervals, index, s1, axis=0)
        intervals[index+1] = s2
        labels = np.insert(labels, index, labels[index])
    return intervals, labels

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
