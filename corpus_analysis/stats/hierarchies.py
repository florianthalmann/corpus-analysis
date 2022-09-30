from collections import defaultdict, Counter
import numpy as np
import numpy_indexed as npi
from ..features import to_multinomial
from ..alignment.smith_waterman import smith_waterman
from ..structure.grammars import pcfg_from_tree, description_length, pcfg_dl
from ..util import flatten
from .util import entropy, subsequences

#true if only one level contains repetition
def auto_labeled(hierarchy):
    hierarchy = remove_silence(hierarchy)
    if min([len(h) for h in hierarchy[0]]) == 0:#empty layers are also auto-labeled...
        return False
    h = to_int_labels(hierarchy)[1]
    merged = lambda l: l[np.hstack(([0], np.where(np.diff(l) != 0)[0]+1))]
    all_diff = lambda l: len(np.unique(l)) == len(merged(l))
    return all_diff(h[0]) != all_diff(h[1])
    #return all_diff(h[1]) or all_diff(h[0]) #top level repeats, bottom not

def remove_silence(hierarchy):
    rem = [tuple(zip(*[s for s in zip(*l) if s[1] != 'Silence']))
        for l in zip(*hierarchy)]
    rem = [r if len(r) > 0 else (np.array([]), []) for r in rem]#deal with entirely silent levels
    rem = tuple(zip(*rem))
    return (tuple([np.vstack(r) if len(r) > 0 else np.array([]) for r in rem[0]]), rem[1])

#def replace_second_silence()

#proportion of label reusage
def repetitiveness(hierarchy):
    h = to_int_labels(hierarchy)[1]
    return np.mean([(len(l)-len(np.unique(l))) / (len(l)-1) if len(l) > 1 else 1
        for l in h])

#avg num sections per level
def complexity(hierarchy):
    return np.mean([len(l) for l in hierarchy[0]])

#avg section duration per level as proportion of total duration
def simplicity(hierarchy):
    return np.mean(np.hstack([[i[1]-i[0] for i in l] for l in hierarchy[0]])) \
        / np.max(np.concatenate(hierarchy[0]))
    # return np.mean([np.mean([i[1]-i[0] for i in l]) for l in hierarchy[0]]) \
    #     / np.max(np.concatenate(hierarchy[0]))

#num sections in original vs in monotonic tree (1 if original is monotonic)
def treeness(hierarchy):
    mono = make_monotonic(hierarchy)
    return sum([len(l) for l in hierarchy[0]]) / sum([len(l) for l in mono[0]])

#mean increase in information content between levels
def salience_time(hierarchy, n=1):
    hierarchy = to_int_labels(hierarchy)
    ng = [to_int(ngrams(l, n)) for l in hierarchy[1]] if n>1 else hierarchy[1]
    entropies = [entropy(l) for l in ng]
    return np.mean(np.diff(entropies))

def ngrams(a, n):
    return [str(g) for g in subsequences(a, n)]

def dlength(hierarchy):
    #hierarchy = list(hierarchy[0]), list(hierarchy[1])
    #add bottom level
    # maxtime = np.max(flatten(hierarchy[0]))
    # beats = beats[np.where(beats <= maxtime)]
    # hierarchy[0].append(np.array(list(zip(beats[:-1], beats[1:]))))
    # hierarchy[1].append(np.array([str(i) for i in range(len(beats)-1)]))
    #int labels and child dict
    hierarchy = to_int_labels(hierarchy)
    hierarchy = make_monotonic(hierarchy)
    #prodrules from this
    #print(hierarchy[1], child_dict)
    tree = to_tree(hierarchy)
    if tree[0] != 'S': tree[0] = 'S'
    pcfg = pcfg_from_tree(tree)
    #description length relative to length of bottom level
    # bottom_labels = hierarchy[1][-1]
    # return description_length(pcfg, [bottom_labels]) / len(bottom_labels)
    #size of grammar relative to size of tree
    return pcfg_dl(pcfg) / tree_size(tree)

def add_top_level(hierarchy, label='TOP'):
    if len(hierarchy[0]) > 1:
        hierarchy = list(hierarchy[0]), list(hierarchy[1])
        flatint = flatten(list(hierarchy[0]))
        hierarchy[0].insert(0, np.array([[np.min(flatint), np.max(flatint)]]))
        hierarchy[1].insert(0, [label])
        hierarchy = tuple(hierarchy[0]), tuple(hierarchy[1])
    return hierarchy

#boolean monotonicity: all interval times of lower levels are contained in higher levels
def monotonicity(hierarchy):
    ivls = hierarchy[0]
    return all([set(np.unique(ivls[i])) >= set(np.unique(ivls[i-1]))
        for i in range(1, len(ivls))])

#mcfee/kinnard monotonicity
def label_monotonicity(params):
    hierarchy, beats = params
    return pairwise_recalls(beatwise_ints(hierarchy, beats))

def label_monotonicity2(params):
    hierarchy, beats = params
    return pairwise_recalls3(beatwise_ints(hierarchy, beats))

#monotonicity without dependencies between different same-label areas
#paper: interval monotonicity
def interval_monotonicity(params):
    hierarchy, beats = params
    #reindex labels ignoring repetition
    hierarchy = hierarchy[0], [np.arange(len(l)) for l in hierarchy[1]]
    labels = beatwise_ints(hierarchy, beats)
    #labels = np.array([relabel_adjacent(l) for l in labels])
    return pairwise_recalls(labels)

def strict_transitivity(params):
    hierarchy, beats = params
    return transitivity(hierarchy, beats, num_identical_pairs)

def order_transitivity(params, delta=2):
    hierarchy, beats = params
    return transitivity(hierarchy, beats, lambda c: num_similar(c, delta))

#of all parent pairs with same labels, how many child sequences are similar
def transitivity(hierarchy, beats, sim_func, ignore_self=True):
    child_dict = to_child_dict(to_tree(hierarchy, beats))
    if ignore_self:
        child_dict = {k:[c for c in cs if len(c) > 1 or c[0] != k]
            for k,cs in child_dict.items()}
    num_conns = num_connections([len(cs) for cs in child_dict.values()])
    if num_conns == 0: return 1
    return sum([sim_func(c) for c in child_dict.values()]) / num_conns

#of all possible pairs of parents, how many have either different label or similar child sequences
def transitivity2(hierarchy, beats, sim_func):
    child_dict = to_child_dict(to_tree(hierarchy, beats))
    num_conns = num_connections([len(cs) for cs in child_dict.values()])
    if num_conns == 0: return 1
    num_sim = sum([sim_func(c) for c in child_dict.values()])
    total = num_connections([sum([len(cs) for cs in child_dict.values()])])
    return (num_sim+(total-num_conns)) / total

#sequence of all occurring children for each parent
def to_child_dict(tree, child_dict=None):
    if not child_dict: child_dict = defaultdict(list)#default param didn't work
    for c in tree[1:]:
        if len(c) > 1:
            child_dict[c[0]].append([cc[0] for cc in c[1:]])
            to_child_dict(c, child_dict)
    return child_dict

def num_identical_pairs(list):
    list = [str(l) for l in list]
    return num_connections([c[1] for c in Counter(list).most_common()])

def num_similar(list, delta):
    num = 0
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            max_len = max(len(list[i]), len(list[j]))
            #print(list[i], list[j], smith_waterman(list[i], list[j])[0])
            if len(smith_waterman(list[i], list[j])[0]) >= max_len-delta:
                num += 1
    return num

def num_connections(group_sizes):
    return sum([n*(n-1)/2 for n in group_sizes])

#of all similarly labeled pairs on each level, how many have the same parent
def pairwise_recalls(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    return np.mean([len(same[i].intersection(same[i-1])) / len(same[i])
        for i in range(1, len(same))])

#of all possible pairs on each level, how many are either labeled differently or have the same parent
def pairwise_recalls2(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    nc = [num_connections([len(l)]) for l in labels]
    smnc = [n-len(s) for s,n in zip(same,nc)]
    return np.mean([(len(same[i].intersection(same[i-1]))+smnc[i]) / nc[i]
        for i in range(1, len(same))])

#of all similarly labeled pairs on each level, how many have the same parent or have a parent with the same label (appearing alone)
def pairwise_recalls3(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [np.array(list(zip(*np.nonzero(s)))) for s in same]
    #for each pair: label and both parent labels
    values = [[np.hstack(([labels[i][p[0]]], labels[i-1][p]))
        for p in same[i]] for i in range(1, len(same))]
    return np.mean([np.mean([1 if len(np.unique(v)) < 3 else 0 for v in vs]) for vs in values])

def to_tree(hierarchy, beats=None):
    #beat ids instead of times
    hierarchy = list(hierarchy[0]), list(hierarchy[1])
    # print(hierarchy[0][0], beats)
    # print(np.reshape(npi.indices(beats, np.hstack(hierarchy[0][0])), (-1,2)))
    # hierarchy[0] = [np.reshape(npi.indices(beats, np.hstack(l)), (-1,2))
    #     for l in hierarchy[0]]
    # print(hierarchy)
    #add top node
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
            #keep nodes that are not yet children
            tree = [n for n in tree if not (i[0] <= n[0][0] and n[0][1] <= i[1])]
            tree.append((i,l,children))
    return to_label_tree(tree[0])

#e.g. [0,[1,1],[2]] (from [([0,3],0,[([0,2],1,[]),([2,3],2,[])])] shape)
def to_label_tree(tree):
    #t = [tree[1]] if len(tree[2]) > 0 else [tree[1] for i in range(tree[0][0], tree[0][1])]
    return [tree[1]] + [to_label_tree(c) for c in tree[2]]

def tree_size(tree):
    return 1 + sum([tree_size(t) for t in tree[1:]])

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

def to_int(a):
    unique = np.unique(a, axis=0)
    unique_index = lambda f: np.where(unique == f)[0][0]
    return np.array([unique_index(f) for f in a])

def relabel_adjacent(labels):
    sp = np.split(labels, np.where(np.diff(labels) != 0)[0]+1)
    return np.hstack([np.repeat(i,len(s)) for i,s in enumerate(sp)])

#print(pairwise_recalls([[0,0,0,1,1,1],[2,2,3,3,4,4]]))
#print(recursive_transitivity([0,[1,[2],[3]],[1,[2]]]))#[0,[[1,[[2,[]],[3,[]]]]]]))#,(1,[4,5]),(0,[2]),(1,[4,5])]))
#print(to_child_dict([0,[1,[2],[3]],[1,[2]]]))
#print(num_similar([[0,1],[0,1,2],[0,1,2,3]], 1))
#print(strict_transitivity([0,[1,[2,[4],[5],[6]],[3]],[1,[2,[4],[5]],[3]], [2,[4]]]))
#print(order_transitivity([0,[1,[2,[4],[5],[6]],[3]],[1,[2,[4],[5]],[3]], [2,[4]]]))
#print(to_tree(([[[0,3],[3,6]],[[0,2],[2,4],[4,6]]], [[0,1],[2,3,4]])))
#print(to_tree(([[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]],[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]]], [[0,0,0,1,1,1],[2,2,3,3,4,4]])))
# hierarchy = ([np.array([[0,3],[3,6]]),np.array([[0,2],[2,4],[4,6]])], [np.array([0,1]),np.array([2,0,4])])
# # #print(label_monotonicity(hierarchy, [0,1,2,3,4,5,6]))
# # print(salience_time(hierarchy))
# print(dlength(hierarchy))